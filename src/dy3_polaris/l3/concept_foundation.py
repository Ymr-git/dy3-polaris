"""Canonical concept foundation for the Dy3+ teaching domain.

This module is deliberately separate from the existing entity, KP and retrieval
models.  It provides stable concept identity and honest mappings to current L3
assets without changing any runtime retrieval or API behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from dy3_polaris.l2.kp_catalog import NEW_KP_NAMES

if TYPE_CHECKING:
    from dy3_polaris.l3.models import DocumentChunk, KnowledgeEntity
    from dy3_polaris.l3.store import KnowledgeStore


class ConceptType(str, Enum):
    """Small closed type set for the initial canonical concept catalogue."""

    SCIENTIFIC_CONCEPT = "scientific_concept"
    MECHANISM = "mechanism"
    MATERIAL = "material"
    PARAMETER = "parameter"
    CHARACTERIZATION = "characterization"
    EVALUATION_METRIC = "evaluation_metric"
    APPLICATION = "application"


class MappingAssetType(str, Enum):
    KP = "kp"
    ENTITY = "entity"
    CHUNK = "chunk"


class MappingStatus(str, Enum):
    CURATED = "curated"
    CANDIDATE = "candidate"


class EvidenceRole(str, Enum):
    """Possible evidence roles; automatic mapping only emits MENTIONS."""

    MENTIONS = "mentions"
    DEFINES = "defines"
    SUPPORTS = "supports"
    LIMITS = "limits"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True, slots=True)
class KnowledgeConcept:
    concept_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    concept_type: ConceptType
    description: str
    domain: str
    source_origin: str
    related_kps: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConceptMapping:
    mapping_id: str
    concept_id: str
    asset_type: MappingAssetType
    asset_id: str
    source_reference: str
    matched_alias: str
    confidence: float
    status: MappingStatus


@dataclass(frozen=True, slots=True)
class ConceptEvidenceMapping:
    mapping_id: str
    concept_id: str
    chunk_id: str
    document_id: str
    claim_scope: str
    evidence_role: EvidenceRole
    confidence: float
    status: MappingStatus
    source_reference: str
    matched_alias: str
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ConceptFoundation:
    """Read-only result of mapping canonical concepts to current assets."""

    concepts: Mapping[str, KnowledgeConcept]
    mappings: tuple[ConceptMapping, ...]
    evidence_mappings: tuple[ConceptEvidenceMapping, ...]

    def get_concept(self, concept_id: str) -> KnowledgeConcept | None:
        return self.concepts.get(concept_id)

    def mappings_for(self, concept_id: str) -> tuple[ConceptMapping, ...]:
        return tuple(item for item in self.mappings if item.concept_id == concept_id)

    def evidence_for(self, concept_id: str) -> tuple[ConceptEvidenceMapping, ...]:
        return tuple(item for item in self.evidence_mappings if item.concept_id == concept_id)

    def stats(self) -> dict[str, int]:
        return {
            "concepts": len(self.concepts),
            "kp_mappings": sum(m.asset_type is MappingAssetType.KP for m in self.mappings),
            "entity_mappings": sum(m.asset_type is MappingAssetType.ENTITY for m in self.mappings),
            "chunk_candidates": sum(m.asset_type is MappingAssetType.CHUNK for m in self.mappings),
            "evidence_candidates": len(self.evidence_mappings),
            "concepts_with_evidence_candidates": len(
                {item.concept_id for item in self.evidence_mappings}
            ),
        }


_DOMAIN = "dy3-green-healthy-lighting"
_ORIGIN = "R06B curated Dy3+ core concept catalogue"


def _concept(
    slug: str,
    name: str,
    concept_type: ConceptType,
    description: str,
    aliases: Iterable[str],
    related_kps: Iterable[str] = (),
) -> KnowledgeConcept:
    return KnowledgeConcept(
        concept_id=f"concept:dy3:{slug}",
        canonical_name=name,
        aliases=tuple(dict.fromkeys((name, *aliases))),
        concept_type=concept_type,
        description=description,
        domain=_DOMAIN,
        source_origin=_ORIGIN,
        related_kps=tuple(dict.fromkeys(related_kps)),
    )


# Explicit IDs are intentional: concept identity does not depend on entity or chunk IDs.
_CORE_CONCEPTS: tuple[KnowledgeConcept, ...] = (
    _concept("rare-earth-electron-configuration", "稀土离子电子构型", ConceptType.SCIENTIFIC_CONCEPT, "稀土离子基态与激发态电子排布的描述。", ("电子构型", "rare-earth electron configuration", "electronic configuration"), ("1.1.1",)),
    _concept("four-f-shell", "4f壳层", ConceptType.SCIENTIFIC_CONCEPT, "稀土离子中受外层轨道屏蔽的4f电子壳层。", ("4f shell", "4f 壳层"), ("1.1.2",)),
    _concept("dy3-energy-level-structure", "Dy3+能级结构", ConceptType.SCIENTIFIC_CONCEPT, "Dy3+离子相关光谱项及其能级分布。", ("Dy³⁺能级结构", "Dy3+能级", "Dy3+ energy levels", "Dy3+ energy level"), ("1.2.1", "2.1.1")),
    _concept("ls-coupling-selection-rules", "L-S耦合与跃迁选择定则", ConceptType.SCIENTIFIC_CONCEPT, "用于组织光谱项并判断光学跃迁允许性的角动量耦合与选择定则。", ("LS耦合", "L-S coupling", "selection rules", "跃迁选择定则"), ("1.2.2",)),
    _concept("four-f-four-f-transition", "4f-4f跃迁", ConceptType.MECHANISM, "同一稀土离子4f电子组态内能级之间的跃迁。", ("4f–4f跃迁", "4f-4f transition"), ("2.1.1",)),
    _concept("four-f-five-d-transition", "4f-5d宽带跃迁", ConceptType.MECHANISM, "稀土离子4f与5d组态之间通常呈宽带特征的光学跃迁。", ("4f-5d跃迁", "4f–5d transition", "4f-5d transition", "broadband transition"), ("2.1.2",)),
    _concept(
        "dy3-blue-emission",
        "Dy3+蓝光发射",
        ConceptType.SCIENTIFIC_CONCEPT,
        "Dy3+可见发射中的蓝光发射分量。",
        (
            "Dy³⁺蓝光发射",
            "Dy3+ blue emission",
            "blue emission",
            "4F9/2 to 6H15/2",
            "4F9/2 → 6H15/2",
            "4F9/2 - 6H15/2",
            "黄蓝双发射",
            "yellow blue dual emission",
        ),
        ("2.1.1", "3.3.2"),
    ),
    _concept(
        "dy3-yellow-emission",
        "Dy3+黄光发射",
        ConceptType.SCIENTIFIC_CONCEPT,
        "Dy3+可见发射中的黄光发射分量。",
        (
            "Dy³⁺黄光发射",
            "Dy3+ yellow emission",
            "yellow emission",
            "4F9/2 to 6H13/2",
            "4F9/2 → 6H13/2",
            "4F9/2 - 6H13/2",
            "黄蓝双发射",
            "yellow blue dual emission",
        ),
        ("2.1.1", "3.3.2"),
    ),
    _concept("yellow-blue-ratio", "黄蓝发射强度比", ConceptType.PARAMETER, "Dy3+黄光与蓝光发射强度的相对比值。", ("黄蓝比", "Y/B ratio", "yellow-to-blue ratio"), ("3.3.2",)),
    _concept("crystal-field", "晶体场", ConceptType.SCIENTIFIC_CONCEPT, "发光中心局部配位环境产生的静电场描述。", ("crystal field",), ("1.3.1", "3.2.2")),
    _concept("stark-splitting", "Stark能级劈裂", ConceptType.MECHANISM, "晶体场作用下离子能级产生的子能级分裂。", ("Stark splitting", "Stark 劈裂"), ("1.3.2",)),
    _concept("judd-ofelt-theory", "Judd-Ofelt理论", ConceptType.SCIENTIFIC_CONCEPT, "用于描述稀土离子光谱跃迁强度的理论框架。", ("Judd–Ofelt理论", "J-O theory"), ("2.2.1",)),
    _concept("radiative-transition-rate", "辐射跃迁速率", ConceptType.PARAMETER, "激发态通过发射光子发生跃迁的速率。", ("radiative transition rate",), ("2.2.2",)),
    _concept("fluorescence-lifetime", "荧光寿命", ConceptType.PARAMETER, "发光中心激发态布居衰减的时间尺度。", ("发光寿命", "fluorescence lifetime"), ("2.2.2", "5.2.2")),
    _concept("energy-transfer", "能量传递", ConceptType.MECHANISM, "激发能在发光中心或不同中心之间转移的过程。", ("energy transfer",), ("2.3.1",)),
    _concept("cross-relaxation", "交叉弛豫", ConceptType.MECHANISM, "相邻离子通过能量交换同时改变能级占据的过程。", ("cross relaxation", "cross-relaxation"), ("2.3.1", "2.3.2")),
    _concept("concentration-quenching", "浓度猝灭", ConceptType.MECHANISM, "发光中心浓度增大后由能量迁移等通道引起的发光效率下降。", ("concentration quenching",), ("2.3.2",)),
    _concept("thermal-quenching", "热猝灭", ConceptType.MECHANISM, "温度升高导致非辐射失活增强并使发光减弱的现象。", ("temperature quenching", "thermal quenching"), ("2.3.3",)),
    _concept("nonradiative-relaxation", "非辐射弛豫", ConceptType.MECHANISM, "激发能不以光子形式释放的弛豫过程。", ("non-radiative relaxation", "nonradiative relaxation"), ("2.3.3",)),
    _concept("phonon-energy", "基质声子能量", ConceptType.PARAMETER, "宿主晶格振动量子对应的能量尺度。", ("phonon energy", "声子能量"), ("2.3.3", "3.1.1")),
    _concept("host-lattice", "发光基质晶格", ConceptType.MATERIAL, "承载稀土发光中心并决定局部结构环境的宿主晶格。", ("host lattice", "host material"), ("3.1.1", "3.1.2", "3.1.3")),
    _concept("local-coordination", "局部配位环境", ConceptType.SCIENTIFIC_CONCEPT, "发光中心周围配位原子的数量、对称性和空间结构。", ("local coordination environment", "格位对称性"), ("3.2.2",)),
    _concept("charge-compensation", "电荷补偿", ConceptType.MECHANISM, "异价掺杂时维持晶格电中性的缺陷或共掺杂机制。", ("charge compensation",), ("3.2.1",)),
    _concept("defects-traps", "缺陷与陷阱态", ConceptType.SCIENTIFIC_CONCEPT, "晶格缺陷形成的局域电子或空穴俘获状态。", ("defect states", "trap states", "陷阱态"), ("3.4.2",)),
    _concept("upconversion-quantum-cutting", "上转换与量子剪裁", ConceptType.MECHANISM, "多光子能量转换中的上转换发光与量子剪裁概念。", ("上转换", "量子剪裁", "upconversion", "quantum cutting"), ("3.4.1",)),
    _concept("nanomaterial-core-shell", "纳米材料表面效应与核壳结构", ConceptType.MATERIAL, "纳米尺寸下表面态及核壳包覆对发光与能量损失的影响。", ("核壳结构", "表面效应", "core-shell structure", "core shell nanoparticle", "surface quenching"), ("3.4.3",)),
    _concept("quantum-efficiency", "发光量子效率", ConceptType.EVALUATION_METRIC, "发射光子数相对于吸收或激发光子数的效率指标。", ("quantum efficiency", "量子效率"), ("3.3.1", "5.2.3")),
    _concept("excitation-spectrum", "激发光谱", ConceptType.CHARACTERIZATION, "监测指定发射时随激发波长变化获得的光谱。", ("excitation spectrum",), ("3.3.3", "5.2.1")),
    _concept("emission-spectrum", "发射光谱", ConceptType.CHARACTERIZATION, "固定激发条件下记录发射强度随波长变化的光谱。", ("emission spectrum",), ("5.2.1",)),
    _concept("cie-chromaticity", "CIE色坐标", ConceptType.EVALUATION_METRIC, "在CIE色度系统中表征光色位置的坐标。", ("CIE chromaticity", "色坐标"), ("3.3.2", "5.2.6")),
    _concept("correlated-color-temperature", "相关色温", ConceptType.EVALUATION_METRIC, "以最接近黑体轨迹的颜色温度描述白光色貌的指标。", ("CCT", "correlated color temperature", "色温"), ("6.1.2",)),
    _concept("color-rendering-index", "显色指数", ConceptType.EVALUATION_METRIC, "评价光源对物体颜色再现能力的指标体系。", ("CRI", "color rendering index"), ("6.1.2",)),
    _concept("blue-light-hazard", "蓝光危害", ConceptType.EVALUATION_METRIC, "依据光谱加权评估短波可见光视网膜光化学风险的概念。", ("blue light hazard", "blue-light hazard", "蓝光风险", "光生物安全", "photobiological safety"), ("6.2.1",)),
    _concept("white-led", "白光LED", ConceptType.APPLICATION, "产生白光输出的发光二极管照明技术。", ("white LED", "白光发光二极管"), ("6.1.1",)),
    _concept("single-phase-white-phosphor", "单基质白光荧光粉", ConceptType.MATERIAL, "在单一宿主体系中形成白光发射的荧光材料。", ("single-phase white phosphor", "单相白光荧光粉"), ("6.3.1",)),
    _concept("solid-state-synthesis", "固相烧结法", ConceptType.SCIENTIFIC_CONCEPT, "通过固体原料混合与高温反应制备发光材料的合成路线。", ("高温固相法", "固相法", "solid-state method", "solid state reaction"), ("4.1.1",)),
    _concept("coprecipitation-synthesis", "共沉淀法", ConceptType.SCIENTIFIC_CONCEPT, "使多种组分在溶液中共同沉淀形成前驱体的合成路线。", ("chemical co-precipitation", "coprecipitation", "co-precipitation"), ("4.1.2",)),
    _concept("sol-gel-synthesis", "溶胶-凝胶法", ConceptType.SCIENTIFIC_CONCEPT, "经由溶胶到凝胶网络并经后续热处理获得材料的合成路线。", ("溶胶凝胶法", "sol-gel method", "sol gel"), ("4.1.3",)),
    _concept("hydrothermal-solvothermal-synthesis", "水热/溶剂热法", ConceptType.SCIENTIFIC_CONCEPT, "在密闭体系中以水或其他溶剂进行高温高压反应的合成路线。", ("水热法", "溶剂热法", "hydrothermal", "solvothermal"), ("4.1.4",)),
    _concept("calcination-temperature-crystallinity", "焙烧温度与结晶度", ConceptType.PARAMETER, "焙烧温度对物相形成、晶粒与结晶度的影响。", ("焙烧温度", "烧结温度", "calcination temperature", "crystallinity"), ("4.2.1",)),
    _concept("reducing-atmosphere-valence", "还原气氛与价态控制", ConceptType.PARAMETER, "合成气氛对发光中心价态及缺陷平衡的影响。", ("还原气氛", "价态控制", "reducing atmosphere", "valence control"), ("4.2.2",)),
    _concept("flux-grain-morphology", "助熔剂与晶粒形貌调控", ConceptType.PARAMETER, "助熔剂对晶体生长、晶粒形貌和烧结过程的影响。", ("助熔剂", "晶粒形貌", "flux", "grain morphology"), ("4.2.3",)),
    _concept("process-defect-control", "工艺参数与缺陷控制", ConceptType.MECHANISM, "合成和热处理参数对晶格缺陷类型与浓度的调控。", ("工艺参数", "缺陷控制", "process parameters", "defect control"), ("4.2.5",)),
    _concept("scale-up-batch-consistency", "规模放大与批次一致性", ConceptType.APPLICATION, "材料制备由小试向更大规模转换时的批次稳定性与一致性问题。", ("规模化制备", "批次一致性", "scale-up", "batch consistency"), ("4.2.6",)),
    _concept("xrd", "X射线衍射", ConceptType.CHARACTERIZATION, "用于分析材料物相与晶体结构的衍射表征方法。", ("XRD", "X-ray diffraction"), ("5.1.1",)),
    _concept("electron-microscopy", "SEM/TEM形貌表征", ConceptType.CHARACTERIZATION, "利用扫描或透射电子显微镜表征颗粒尺寸、形貌与微结构。", ("SEM", "TEM", "scanning electron microscopy", "transmission electron microscopy", "电镜形貌"), ("5.1.2",)),
    _concept("photoluminescence-spectroscopy", "光致发光光谱", ConceptType.CHARACTERIZATION, "利用光激发记录材料发光响应的光谱表征方法。", ("PL spectroscopy", "PL光谱"), ("5.2.1",)),
    _concept("lifetime-measurement", "荧光寿命测量", ConceptType.CHARACTERIZATION, "测量激发停止后发光衰减曲线并拟合寿命的方法。", ("lifetime measurement",), ("5.2.2",)),
    _concept("integrating-sphere", "积分球量子效率测量", ConceptType.CHARACTERIZATION, "利用积分球进行绝对光子收支测量的量子效率表征方法。", ("integrating sphere", "绝对法量子效率"), ("5.2.3",)),
    _concept("thermal-stability-measurement", "热稳定性T50测试", ConceptType.CHARACTERIZATION, "通过变温发光数据表征材料热稳定性及T50等指标的测试。", ("热稳定性测试", "T50", "temperature-dependent luminescence", "thermal stability test"), ("5.2.4",)),
    _concept("icp-oes-doping-quantification", "ICP-OES掺杂浓度定量", ConceptType.CHARACTERIZATION, "利用电感耦合等离子体发射光谱对材料元素含量进行定量。", ("ICP-OES", "inductively coupled plasma optical emission spectroscopy", "掺杂浓度定量"), ("5.2.5",)),
    _concept("doping-concentration", "Dy3+掺杂浓度", ConceptType.PARAMETER, "Dy3+发光中心在宿主材料中的含量或占位比例。", ("Dy³⁺掺杂浓度", "doping concentration"), ("4.2.4", "2.3.2")),
    _concept("green-healthy-lighting", "绿色健康照明", ConceptType.APPLICATION, "同时关注能效、光品质和光生物安全的照明应用语境。", ("green healthy lighting", "健康照明"), ("6.2.1", "6.2.2", "6.3.2")),
)


def canonical_concepts() -> tuple[KnowledgeConcept, ...]:
    """Return the immutable, curated initial catalogue."""

    return _CORE_CONCEPTS


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:{sha256(payload).hexdigest()[:20]}"


def _normalise(value: str) -> str:
    return re.sub(r"[\s\-–—_]+", "", value).casefold().replace("³⁺", "3+")


def _matched_alias(text: str, aliases: Iterable[str]) -> str | None:
    normalised_text = _normalise(text)
    for alias in sorted(aliases, key=len, reverse=True):
        if len(_normalise(alias)) >= 3 and _normalise(alias) in normalised_text:
            return alias
    return None


def _entity_alias_match(entity: KnowledgeEntity, concept: KnowledgeConcept) -> str | None:
    if {"kp", "knowledge_point", "fact"}.intersection(entity.tags):
        return None
    entity_names = (entity.name, *entity.aliases)
    concept_names = { _normalise(value): value for value in concept.aliases }
    for value in entity_names:
        if _normalise(value) in concept_names:
            return concept_names[_normalise(value)]
    return None


def _chunk_candidates(
    store: KnowledgeStore,
    concept: KnowledgeConcept,
    *,
    limit: int,
) -> tuple[tuple[DocumentChunk, str], ...]:
    found: dict[str, tuple[DocumentChunk, str]] = {}
    for alias in sorted(concept.aliases, key=len, reverse=True):
        if len(_normalise(alias)) < 3:
            continue
        for chunk, _retrieval_score in store.search_text(alias, top_k=max(limit * 2, 5)):
            matched = _matched_alias(chunk.content, concept.aliases)
            if matched is not None:
                found.setdefault(chunk.chunk_id, (chunk, matched))
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    return tuple(found.values())


def build_concept_foundation(
    store: KnowledgeStore | None = None,
    *,
    max_evidence_per_concept: int = 3,
) -> ConceptFoundation:
    """Build a read-only mapping view over current knowledge assets.

    KP mappings are curated against the existing L2 catalogue. Entity mappings
    require exact canonical-name/alias equality. Chunk matches remain unverified
    candidates and only assert that a term is mentioned.
    """

    if max_evidence_per_concept < 0:
        raise ValueError("max_evidence_per_concept must be non-negative")

    concepts = {item.concept_id: item for item in canonical_concepts()}
    mappings: list[ConceptMapping] = []
    evidence_mappings: list[ConceptEvidenceMapping] = []

    for concept in concepts.values():
        for kp_id in concept.related_kps:
            if kp_id not in NEW_KP_NAMES:
                raise ValueError(f"unknown KP mapping: {kp_id}")
            mappings.append(ConceptMapping(
                mapping_id=_stable_id("concept-map", concept.concept_id, "kp", kp_id),
                concept_id=concept.concept_id,
                asset_type=MappingAssetType.KP,
                asset_id=f"kp-catalog:{kp_id}",
                source_reference="dy3_polaris.l2.kp_catalog.NEW_KP_NAMES",
                matched_alias=NEW_KP_NAMES[kp_id],
                confidence=1.0,
                status=MappingStatus.CURATED,
            ))

        if store is None:
            continue

        for entity in store.entity_store.list_entities(limit=max(store.entity_count(), 1)):
            matched = _entity_alias_match(entity, concept)
            if matched is None:
                continue
            mappings.append(ConceptMapping(
                mapping_id=_stable_id("concept-map", concept.concept_id, "entity", entity.entity_id),
                concept_id=concept.concept_id,
                asset_type=MappingAssetType.ENTITY,
                asset_id=entity.entity_id,
                source_reference=f"KnowledgeStore.entity_store:{entity.entity_id}",
                matched_alias=matched,
                confidence=0.0,
                status=MappingStatus.CANDIDATE,
            ))

        if max_evidence_per_concept == 0:
            continue
        for chunk, matched in _chunk_candidates(
            store, concept, limit=max_evidence_per_concept
        ):
            evidence_id = _stable_id(
                "concept-evidence", concept.concept_id, chunk.document_id, chunk.chunk_id
            )
            source_ref = f"KnowledgeStore.chunk_store:{chunk.chunk_id}"
            evidence_mappings.append(ConceptEvidenceMapping(
                mapping_id=evidence_id,
                concept_id=concept.concept_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                claim_scope=f"term mention only: {matched}",
                evidence_role=EvidenceRole.MENTIONS,
                confidence=0.0,
                status=MappingStatus.CANDIDATE,
                source_reference=source_ref,
                matched_alias=matched,
                section=chunk.section or None,
                page=chunk.page if chunk.page > 0 else None,
            ))
            mappings.append(ConceptMapping(
                mapping_id=_stable_id("concept-map", concept.concept_id, "chunk", chunk.chunk_id),
                concept_id=concept.concept_id,
                asset_type=MappingAssetType.CHUNK,
                asset_id=chunk.chunk_id,
                source_reference=source_ref,
                matched_alias=matched,
                confidence=0.0,
                status=MappingStatus.CANDIDATE,
            ))

    evidence_by_concept: dict[str, list[str]] = {}
    for item in evidence_mappings:
        evidence_by_concept.setdefault(item.concept_id, []).append(item.mapping_id)
    concepts = {
        concept_id: replace(
            concept,
            evidence_refs=tuple(evidence_by_concept.get(concept_id, ())),
        )
        for concept_id, concept in concepts.items()
    }
    return ConceptFoundation(
        concepts=MappingProxyType(concepts),
        mappings=tuple(mappings),
        evidence_mappings=tuple(evidence_mappings),
    )


_CURATED_EVIDENCE_PATH = (
    Path(__file__).resolve().parent / "data" / "curated" / "concept_evidence.json"
)


def load_curated_concept_evidence(
    store: KnowledgeStore,
    *,
    asset_path: str | Path | None = None,
) -> tuple[DocumentChunk, ...]:
    """Load reviewed, source-backed Concept summaries into ``store``.

    The shipped paper chunks remain the primary corpus.  This compact asset is
    a governed projection over those chunks and a small number of authoritative
    public standards: it makes noisy PDF extraction usable without turning the
    projection into an answer template.  Every record must identify its
    canonical Concepts and an independently traceable source URI.  Stable IDs
    make loading idempotent and independent of mutable snapshots.
    """

    from dy3_polaris.l3.models import DocumentChunk

    path = Path(asset_path) if asset_path is not None else _CURATED_EVIDENCE_PATH
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("curated Concept evidence must contain an entries list")

    concept_ids = {concept.concept_id for concept in canonical_concepts()}
    loaded: list[DocumentChunk] = []
    for index, raw in enumerate(payload["entries"]):
        if not isinstance(raw, dict):
            raise ValueError(f"curated evidence entry {index} must be an object")
        entry_id = str(raw.get("entry_id") or "").strip()
        content = str(raw.get("content") or "").strip()
        source_uri = str(raw.get("source_uri") or "").strip()
        source_title = str(raw.get("source_title") or "").strip()
        raw_concept_ids = raw.get("concept_ids")
        if not entry_id or not content or not source_uri or not source_title:
            raise ValueError(
                f"curated evidence entry {index} is missing identity, content, or provenance"
            )
        if not isinstance(raw_concept_ids, list) or not raw_concept_ids:
            raise ValueError(f"curated evidence entry {entry_id} has no Concepts")
        mapped_concepts = tuple(
            dict.fromkeys(str(value).strip() for value in raw_concept_ids if str(value).strip())
        )
        unknown = set(mapped_concepts) - concept_ids
        if unknown:
            raise ValueError(
                f"curated evidence entry {entry_id} references unknown Concepts: {sorted(unknown)}"
            )

        chunk_id = _stable_id("c-curated", entry_id, source_uri)
        existing = store.chunk_store.get_chunk(chunk_id)
        if existing is not None:
            loaded.append(existing)
            continue

        source_chunk_ids = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (raw.get("source_chunk_ids") or [])
                if str(value).strip()
            )
        )
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=f"curated-concept-evidence:{entry_id}",
            content=content,
            chunk_index=index,
            section=str(raw.get("section") or "Reviewed concept evidence"),
            page=0,
            language=str(raw.get("language") or "zh"),
            metadata={
                "source_type": "curated_source_summary",
                "source": source_title,
                "source_title": source_title,
                "source_uri": source_uri,
                "source_chunk_ids": list(source_chunk_ids),
                "concept_ids": list(mapped_concepts),
                "evidence_status": "reviewed",
                "claim_scope": str(raw.get("claim_scope") or "source-bounded summary"),
                "curation_version": str(payload.get("version") or "unknown"),
            },
        )
        store.chunk_store.add_chunk(chunk)
        loaded.append(chunk)
    return tuple(loaded)


__all__ = [
    "ConceptEvidenceMapping",
    "ConceptFoundation",
    "ConceptMapping",
    "ConceptType",
    "EvidenceRole",
    "KnowledgeConcept",
    "MappingAssetType",
    "MappingStatus",
    "build_concept_foundation",
    "canonical_concepts",
    "load_curated_concept_evidence",
]
