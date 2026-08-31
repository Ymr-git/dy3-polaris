"""Controlled scientific relation network over the R-06B concept foundation.

The network is internal to L3.  It does not alter the existing KnowledgeStore,
retrieval pipeline, Agent contracts, learner state, API, or frontend.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

from dy3_polaris.l3.concept_foundation import (
    ConceptFoundation,
    build_concept_foundation,
)


class ConceptRelationType(str, Enum):
    """Minimal closed relation vocabulary for the R-06C network."""

    PREREQUISITE_OF = "prerequisite_of"
    EXPLAINS = "explains"
    CAUSES = "causes"
    AFFECTS = "affects"
    EVALUATED_BY = "evaluated_by"
    APPLIED_IN = "applied_in"


class RelationSource(str, Enum):
    """Origin of a relation assertion, not a scientific evidence score."""

    CURATED = "curated"
    EXISTING_GRAPH = "existing_graph"
    DERIVED = "derived"


class RelationStatus(str, Enum):
    CURATED = "curated"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class ConceptRelation:
    relation_id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: ConceptRelationType
    description: str
    source: RelationSource
    confidence: float
    status: RelationStatus
    source_reference: str

    def __post_init__(self) -> None:
        if not self.relation_id.startswith("relation:dy3:"):
            raise ValueError("relation_id must use the relation:dy3: namespace")
        if self.source_concept_id == self.target_concept_id:
            raise ValueError("self relations are not allowed")
        if not isinstance(self.relation_type, ConceptRelationType):
            raise ValueError("relation_type must be a ConceptRelationType")
        if not isinstance(self.source, RelationSource):
            raise ValueError("source must be a RelationSource")
        if not isinstance(self.status, RelationStatus):
            raise ValueError("status must be a RelationStatus")
        if not self.description.strip():
            raise ValueError("relation description is required")
        if not self.source_reference.strip():
            raise ValueError("relation source_reference is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("relation confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ConceptPath:
    path_id: str
    concept_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.concept_ids) < 1:
            raise ValueError("a concept path requires at least one concept")
        if len(self.concept_ids) != len(self.relation_ids) + 1:
            raise ValueError("concept and relation counts do not form a path")

    @property
    def source_concept_id(self) -> str:
        return self.concept_ids[0]

    @property
    def target_concept_id(self) -> str:
        return self.concept_ids[-1]

    @property
    def hops(self) -> int:
        return len(self.relation_ids)


@dataclass(frozen=True, slots=True)
class ConceptRelationNetwork:
    """Validated, read-only relation graph over a ConceptFoundation."""

    foundation: ConceptFoundation
    relations: tuple[ConceptRelation, ...]
    _relations_by_id: Mapping[str, ConceptRelation] = field(init=False, repr=False)
    _outgoing: Mapping[str, tuple[ConceptRelation, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        known = set(self.foundation.concepts)
        by_id: dict[str, ConceptRelation] = {}
        semantic_keys: set[tuple[str, ConceptRelationType, str]] = set()
        outgoing: dict[str, list[ConceptRelation]] = {concept_id: [] for concept_id in known}

        for relation in self.relations:
            if relation.source_concept_id not in known:
                raise ValueError(f"unknown source concept: {relation.source_concept_id}")
            if relation.target_concept_id not in known:
                raise ValueError(f"unknown target concept: {relation.target_concept_id}")
            if relation.relation_id in by_id:
                raise ValueError(f"duplicate relation_id: {relation.relation_id}")
            semantic_key = (
                relation.source_concept_id,
                relation.relation_type,
                relation.target_concept_id,
            )
            if semantic_key in semantic_keys:
                raise ValueError(f"duplicate semantic relation: {semantic_key}")
            by_id[relation.relation_id] = relation
            semantic_keys.add(semantic_key)
            outgoing[relation.source_concept_id].append(relation)

        object.__setattr__(self, "_relations_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_outgoing", MappingProxyType({
            concept_id: tuple(items) for concept_id, items in outgoing.items()
        }))

    def get_relation(self, relation_id: str) -> ConceptRelation | None:
        return self._relations_by_id.get(relation_id)

    def outgoing(self, concept_id: str) -> tuple[ConceptRelation, ...]:
        self._require_concept(concept_id)
        return self._outgoing[concept_id]

    def find_path(
        self,
        source_concept_id: str,
        target_concept_id: str,
        *,
        max_hops: int = 8,
        allowed_types: frozenset[ConceptRelationType] | None = None,
    ) -> ConceptPath | None:
        """Return the shortest directed curated path, without inventing edges."""

        self._require_concept(source_concept_id)
        self._require_concept(target_concept_id)
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if allowed_types is not None and any(
            not isinstance(item, ConceptRelationType) for item in allowed_types
        ):
            raise ValueError("allowed_types must contain ConceptRelationType values")
        if source_concept_id == target_concept_id:
            return _path((source_concept_id,), ())

        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque([
            (source_concept_id, (source_concept_id,), ()),
        ])
        visited = {source_concept_id}
        while queue:
            current, concept_ids, relation_ids = queue.popleft()
            if len(relation_ids) >= max_hops:
                continue
            for relation in self._outgoing[current]:
                if relation.status is not RelationStatus.CURATED:
                    continue
                if allowed_types is not None and relation.relation_type not in allowed_types:
                    continue
                target = relation.target_concept_id
                next_concepts = (*concept_ids, target)
                next_relations = (*relation_ids, relation.relation_id)
                if target == target_concept_id:
                    return _path(next_concepts, next_relations)
                if target in visited:
                    continue
                visited.add(target)
                queue.append((target, next_concepts, next_relations))
        return None

    def stats(self) -> dict[str, object]:
        participating = {
            concept_id
            for relation in self.relations
            for concept_id in (relation.source_concept_id, relation.target_concept_id)
        }
        return {
            "concepts": len(self.foundation.concepts),
            "relations": len(self.relations),
            "participating_concepts": len(participating),
            "by_type": {
                relation_type.value: sum(
                    item.relation_type is relation_type for item in self.relations
                )
                for relation_type in ConceptRelationType
            },
            "by_source": {
                source.value: sum(item.source is source for item in self.relations)
                for source in RelationSource
            },
        }

    def _require_concept(self, concept_id: str) -> None:
        if concept_id not in self.foundation.concepts:
            raise KeyError(f"unknown concept: {concept_id}")


def _path(concept_ids: tuple[str, ...], relation_ids: tuple[str, ...]) -> ConceptPath:
    payload = "\x1f".join((*concept_ids, *relation_ids)).encode("utf-8")
    return ConceptPath(
        path_id=f"concept-path:dy3:{sha256(payload).hexdigest()[:20]}",
        concept_ids=concept_ids,
        relation_ids=relation_ids,
    )


def _relation(
    slug: str,
    source_slug: str,
    target_slug: str,
    relation_type: ConceptRelationType,
    description: str,
    source: RelationSource,
    source_reference: str,
) -> ConceptRelation:
    return ConceptRelation(
        relation_id=f"relation:dy3:{slug}",
        source_concept_id=f"concept:dy3:{source_slug}",
        target_concept_id=f"concept:dy3:{target_slug}",
        relation_type=relation_type,
        description=description,
        source=source,
        confidence=1.0,
        status=RelationStatus.CURATED,
        source_reference=source_reference,
    )


_KP = "dy3_polaris.l2.kp_catalog.NEW_KP_EDGES"
_R06A = ".ai/R06A_KNOWLEDGE_LAYER_AUDIT_DESIGN.md#7.1"


# Small, explicit network only.  No relation below is inferred from co-occurrence.
_CORE_RELATIONS: tuple[ConceptRelation, ...] = (
    _relation("electron-config-prereq-four-f", "rare-earth-electron-configuration", "four-f-shell", ConceptRelationType.PREREQUISITE_OF, "电子构型是理解4f壳层特征的先修基础。", RelationSource.EXISTING_GRAPH, f"{_KP}:1.1.1:prerequisite_of:1.1.2"),
    # The KP graph relates L-S coupling to the intermediate "atomic spectral
    # term" KP, which is not a canonical Concept in this foundation.  Treating
    # it as a direct prerequisite of Dy3+ energy levels silently skipped that
    # intermediate node and displaced the more fundamental electron-
    # configuration path.  Preserve the scientifically useful explanatory
    # relation without inventing a direct prerequisite edge.
    _relation("ls-rules-explains-energy-level", "ls-coupling-selection-rules", "dy3-energy-level-structure", ConceptRelationType.EXPLAINS, "L-S耦合与选择定则用于解释Dy3+光谱项及能级跃迁的组织方式。", RelationSource.CURATED, _R06A),
    _relation("four-f-prereq-energy-level", "four-f-shell", "dy3-energy-level-structure", ConceptRelationType.PREREQUISITE_OF, "4f轨道特征是理解Dy3+能级结构的先修基础。", RelationSource.EXISTING_GRAPH, f"{_KP}:1.1.2:prerequisite_of:2.1.1"),
    _relation("energy-level-prereq-transition", "dy3-energy-level-structure", "four-f-four-f-transition", ConceptRelationType.PREREQUISITE_OF, "能级结构是解释4f-4f跃迁的先修基础。", RelationSource.CURATED, _R06A),
    _relation("transition-explains-blue", "four-f-four-f-transition", "dy3-blue-emission", ConceptRelationType.EXPLAINS, "Dy3+的4f-4f跃迁解释其蓝光发射分量。", RelationSource.CURATED, _R06A),
    _relation("transition-explains-yellow", "four-f-four-f-transition", "dy3-yellow-emission", ConceptRelationType.EXPLAINS, "Dy3+的4f-4f跃迁解释其黄光发射分量。", RelationSource.CURATED, _R06A),
    _relation("blue-affects-ratio", "dy3-blue-emission", "yellow-blue-ratio", ConceptRelationType.AFFECTS, "蓝光发射强度参与决定黄蓝发射强度比。", RelationSource.CURATED, _R06A),
    _relation("yellow-affects-ratio", "dy3-yellow-emission", "yellow-blue-ratio", ConceptRelationType.AFFECTS, "黄光发射强度参与决定黄蓝发射强度比。", RelationSource.CURATED, _R06A),
    _relation("ratio-affects-cie", "yellow-blue-ratio", "cie-chromaticity", ConceptRelationType.AFFECTS, "黄蓝发射强度比变化会改变合成光的CIE色坐标。", RelationSource.CURATED, _R06A),
    _relation("ratio-affects-cct", "yellow-blue-ratio", "correlated-color-temperature", ConceptRelationType.AFFECTS, "黄蓝光谱配比变化会影响白光的相关色温。", RelationSource.CURATED, _R06A),
    _relation("energy-level-prereq-judd-ofelt", "dy3-energy-level-structure", "judd-ofelt-theory", ConceptRelationType.PREREQUISITE_OF, "能级结构是使用Judd-Ofelt理论分析跃迁强度的前提。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.1.1:prerequisite_of:2.2.1"),
    _relation("judd-explains-radiative-rate", "judd-ofelt-theory", "radiative-transition-rate", ConceptRelationType.EXPLAINS, "Judd-Ofelt强度参数用于解释辐射跃迁速率。", RelationSource.CURATED, _R06A),
    _relation("radiative-rate-affects-lifetime", "radiative-transition-rate", "fluorescence-lifetime", ConceptRelationType.AFFECTS, "辐射跃迁速率是影响激发态寿命的速率分量。", RelationSource.CURATED, _R06A),
    _relation("energy-level-prereq-energy-transfer", "dy3-energy-level-structure", "energy-transfer", ConceptRelationType.PREREQUISITE_OF, "能级匹配是理解能量传递通道的先修条件。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.1.1:prerequisite_of:2.3.1"),
    _relation("lifetime-affects-energy-transfer", "fluorescence-lifetime", "energy-transfer", ConceptRelationType.AFFECTS, "供体激发态寿命会影响可发生能量传递的时间窗口。", RelationSource.CURATED, _R06A),
    _relation("doping-affects-energy-transfer", "doping-concentration", "energy-transfer", ConceptRelationType.AFFECTS, "掺杂浓度改变发光中心间距并影响能量迁移。", RelationSource.CURATED, _R06A),
    _relation("doping-affects-cross-relaxation", "doping-concentration", "cross-relaxation", ConceptRelationType.AFFECTS, "发光中心浓度改变离子间相互作用和交叉弛豫发生机会。", RelationSource.CURATED, _R06A),
    _relation("cross-relaxation-causes-concentration-quenching", "cross-relaxation", "concentration-quenching", ConceptRelationType.CAUSES, "在能级匹配且离子间距足够小时，交叉弛豫可形成浓度猝灭通道。", RelationSource.CURATED, _R06A),
    _relation("doping-causes-concentration-quenching", "doping-concentration", "concentration-quenching", ConceptRelationType.CAUSES, "当Dy3+浓度超过特定基质和工艺条件下的最佳值时，增强的离子相互作用可导致浓度猝灭。", RelationSource.CURATED, _R06A),
    _relation("concentration-quenching-affects-qe", "concentration-quenching", "quantum-efficiency", ConceptRelationType.AFFECTS, "浓度猝灭增加非辐射损失并降低发光量子效率。", RelationSource.CURATED, _R06A),
    _relation("host-affects-local-coordination", "host-lattice", "local-coordination", ConceptRelationType.AFFECTS, "宿主晶格决定Dy3+可占据格位及其局部配位环境。", RelationSource.EXISTING_GRAPH, f"{_KP}:3.1.1|3.1.2|3.1.3:affects:3.2.2"),
    _relation("local-coordination-affects-crystal-field", "local-coordination", "crystal-field", ConceptRelationType.AFFECTS, "局部配位结构与对称性决定发光中心所处晶体场。", RelationSource.EXISTING_GRAPH, f"{_KP}:3.2.2:affects:1.3.1"),
    _relation("crystal-field-explains-stark", "crystal-field", "stark-splitting", ConceptRelationType.EXPLAINS, "晶体场解除部分能级简并并解释Stark劈裂。", RelationSource.CURATED, _R06A),
    _relation("stark-affects-emission-spectrum", "stark-splitting", "emission-spectrum", ConceptRelationType.AFFECTS, "Stark子能级结构会影响发射峰位置和精细结构。", RelationSource.CURATED, _R06A),
    _relation("charge-compensation-affects-defects", "charge-compensation", "defects-traps", ConceptRelationType.AFFECTS, "电荷补偿策略会改变缺陷类型与浓度。", RelationSource.CURATED, _R06A),
    _relation("defects-affect-nonradiative", "defects-traps", "nonradiative-relaxation", ConceptRelationType.AFFECTS, "部分缺陷和陷阱态可引入非辐射复合或能量损失通道。", RelationSource.CURATED, _R06A),
    _relation("phonon-affects-nonradiative", "phonon-energy", "nonradiative-relaxation", ConceptRelationType.AFFECTS, "基质声子能量影响多声子非辐射弛豫的可能性。", RelationSource.CURATED, _R06A),
    _relation("nonradiative-causes-thermal-quenching", "nonradiative-relaxation", "thermal-quenching", ConceptRelationType.CAUSES, "温度升高激活非辐射通道时，非辐射弛豫可导致热猝灭。", RelationSource.CURATED, _R06A),
    _relation("thermal-quenching-affects-qe", "thermal-quenching", "quantum-efficiency", ConceptRelationType.AFFECTS, "热猝灭提高非辐射损失并降低工作温度下的发光效率。", RelationSource.CURATED, _R06A),
    _relation("thermal-quenching-evaluated-by-stability", "thermal-quenching", "thermal-stability-measurement", ConceptRelationType.EVALUATED_BY, "变温发光与T50测试用于评价材料的热猝灭和热稳定性。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.3.3:characterized_by:5.2.4"),
    _relation("calcination-affects-qe", "calcination-temperature-crystallinity", "quantum-efficiency", ConceptRelationType.AFFECTS, "焙烧温度与结晶度会影响发光材料的量子效率。", RelationSource.EXISTING_GRAPH, f"{_KP}:4.2.1:affects:3.3.1"),
    _relation("calcination-evaluated-by-xrd", "calcination-temperature-crystallinity", "xrd", ConceptRelationType.EVALUATED_BY, "焙烧温度引起的结晶度和物相变化可由X射线衍射表征。", RelationSource.EXISTING_GRAPH, f"{_KP}:4.2.1:characterized_by:5.1.1"),
    _relation("reducing-atmosphere-affects-process-defects", "reducing-atmosphere-valence", "process-defect-control", ConceptRelationType.AFFECTS, "合成气氛与价态控制会影响工艺形成的缺陷状态。", RelationSource.EXISTING_GRAPH, f"{_KP}:4.2.2:affects:4.2.5"),
    _relation("process-defects-affect-traps", "process-defect-control", "defects-traps", ConceptRelationType.AFFECTS, "工艺参数对缺陷的调控会改变缺陷与陷阱态。", RelationSource.EXISTING_GRAPH, f"{_KP}:4.2.5:affects:3.4.2"),
    _relation("core-shell-affects-qe", "nanomaterial-core-shell", "quantum-efficiency", ConceptRelationType.AFFECTS, "纳米表面状态与核壳结构会影响非辐射损失和量子效率。", RelationSource.EXISTING_GRAPH, f"{_KP}:3.4.3:affects:3.3.1"),
    _relation("core-shell-evaluated-by-electron-microscopy", "nanomaterial-core-shell", "electron-microscopy", ConceptRelationType.EVALUATED_BY, "SEM/TEM可用于观察纳米材料形貌及核壳微结构。", RelationSource.CURATED, _R06A),
    _relation("flux-morphology-evaluated-by-electron-microscopy", "flux-grain-morphology", "electron-microscopy", ConceptRelationType.EVALUATED_BY, "SEM/TEM可用于评价助熔剂作用下的颗粒和晶粒形貌。", RelationSource.EXISTING_GRAPH, f"{_KP}:4.2.3:characterized_by:5.1.2"),
    _relation("doping-evaluated-by-icp-oes", "doping-concentration", "icp-oes-doping-quantification", ConceptRelationType.EVALUATED_BY, "ICP-OES可用于定量材料中的实际掺杂元素含量。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.3.2:characterized_by:5.2.5"),
    _relation("energy-level-evaluated-by-excitation", "dy3-energy-level-structure", "excitation-spectrum", ConceptRelationType.EVALUATED_BY, "激发光谱提供与可激发能级和跃迁相关的观测信息。", RelationSource.CURATED, _R06A),
    _relation("energy-level-evaluated-by-emission", "dy3-energy-level-structure", "emission-spectrum", ConceptRelationType.EVALUATED_BY, "发射光谱提供发射能级跃迁的观测信息。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.1.1:characterized_by:5.2.1"),
    _relation("blue-evaluated-by-emission", "dy3-blue-emission", "emission-spectrum", ConceptRelationType.EVALUATED_BY, "蓝光发射峰及强度由发射光谱记录。", RelationSource.CURATED, _R06A),
    _relation("yellow-evaluated-by-emission", "dy3-yellow-emission", "emission-spectrum", ConceptRelationType.EVALUATED_BY, "黄光发射峰及强度由发射光谱记录。", RelationSource.CURATED, _R06A),
    _relation("excitation-evaluated-by-pl", "excitation-spectrum", "photoluminescence-spectroscopy", ConceptRelationType.EVALUATED_BY, "光致发光光谱系统可测量激发光谱。", RelationSource.CURATED, _R06A),
    _relation("emission-evaluated-by-pl", "emission-spectrum", "photoluminescence-spectroscopy", ConceptRelationType.EVALUATED_BY, "光致发光光谱系统可测量发射光谱。", RelationSource.CURATED, _R06A),
    _relation("lifetime-evaluated-by-decay", "fluorescence-lifetime", "lifetime-measurement", ConceptRelationType.EVALUATED_BY, "时间分辨衰减测量用于得到荧光寿命。", RelationSource.EXISTING_GRAPH, f"{_KP}:2.2.2:characterized_by:5.2.2"),
    _relation("qe-evaluated-by-integrating-sphere", "quantum-efficiency", "integrating-sphere", ConceptRelationType.EVALUATED_BY, "积分球绝对法用于测量发光量子效率。", RelationSource.EXISTING_GRAPH, f"{_KP}:3.3.1:characterized_by:5.2.3"),
    _relation("host-evaluated-by-xrd", "host-lattice", "xrd", ConceptRelationType.EVALUATED_BY, "X射线衍射用于确认宿主材料物相与晶体结构。", RelationSource.CURATED, _R06A),
    _relation("blue-applied-in-white-phosphor", "dy3-blue-emission", "single-phase-white-phosphor", ConceptRelationType.APPLIED_IN, "Dy3+蓝光发射分量参与单基质白光合成。", RelationSource.CURATED, _R06A),
    _relation("yellow-applied-in-white-phosphor", "dy3-yellow-emission", "single-phase-white-phosphor", ConceptRelationType.APPLIED_IN, "Dy3+黄光发射分量参与单基质白光合成。", RelationSource.CURATED, _R06A),
    _relation("host-applied-in-white-phosphor", "host-lattice", "single-phase-white-phosphor", ConceptRelationType.APPLIED_IN, "宿主晶格是构成单基质白光荧光粉的材料基础。", RelationSource.CURATED, _R06A),
    _relation("white-phosphor-evaluated-by-qe", "single-phase-white-phosphor", "quantum-efficiency", ConceptRelationType.EVALUATED_BY, "量子效率是评价白光荧光粉能量转换性能的指标之一。", RelationSource.CURATED, _R06A),
    _relation("white-phosphor-evaluated-by-cie", "single-phase-white-phosphor", "cie-chromaticity", ConceptRelationType.EVALUATED_BY, "CIE色坐标用于评价白光荧光粉的综合色点。", RelationSource.CURATED, _R06A),
    _relation("white-phosphor-evaluated-by-cct", "single-phase-white-phosphor", "correlated-color-temperature", ConceptRelationType.EVALUATED_BY, "相关色温用于描述白光荧光粉合成光的色貌。", RelationSource.CURATED, _R06A),
    _relation("white-phosphor-evaluated-by-cri", "single-phase-white-phosphor", "color-rendering-index", ConceptRelationType.EVALUATED_BY, "显色指数用于评价器件级光谱的颜色再现能力，不能由单条发射峰直接推定。", RelationSource.CURATED, _R06A),
    _relation("white-phosphor-applied-in-led", "single-phase-white-phosphor", "white-led", ConceptRelationType.APPLIED_IN, "白光荧光粉可作为白光LED的光转换材料。", RelationSource.CURATED, _R06A),
    _relation("white-led-evaluated-by-cct", "white-led", "correlated-color-temperature", ConceptRelationType.EVALUATED_BY, "相关色温描述白光LED的综合色貌。", RelationSource.CURATED, _R06A),
    _relation("white-led-evaluated-by-cri", "white-led", "color-rendering-index", ConceptRelationType.EVALUATED_BY, "显色指数评价白光LED对物体颜色的再现能力。", RelationSource.CURATED, _R06A),
    _relation("white-led-evaluated-by-blue-hazard", "white-led", "blue-light-hazard", ConceptRelationType.EVALUATED_BY, "蓝光危害评价需要白光LED的光谱、辐亮度及暴露条件。", RelationSource.CURATED, _R06A),
    _relation("white-led-applied-in-healthy-lighting", "white-led", "green-healthy-lighting", ConceptRelationType.APPLIED_IN, "白光LED是绿色健康照明的器件应用之一。", RelationSource.CURATED, _R06A),
    _relation("healthy-lighting-evaluated-by-cct", "green-healthy-lighting", "correlated-color-temperature", ConceptRelationType.EVALUATED_BY, "相关色温是健康照明光环境描述的一项指标，但不能单独证明安全。", RelationSource.CURATED, _R06A),
    _relation("healthy-lighting-evaluated-by-cri", "green-healthy-lighting", "color-rendering-index", ConceptRelationType.EVALUATED_BY, "显色指数是健康照明光品质评价的一项指标。", RelationSource.CURATED, _R06A),
    _relation("healthy-lighting-evaluated-by-blue-hazard", "green-healthy-lighting", "blue-light-hazard", ConceptRelationType.EVALUATED_BY, "蓝光危害是健康照明光生物安全边界中的一项评价维度。", RelationSource.CURATED, _R06A),
)


def core_concept_relations() -> tuple[ConceptRelation, ...]:
    """Return the immutable curated R-06C relation set."""

    return _CORE_RELATIONS


def build_concept_relation_network(
    foundation: ConceptFoundation | None = None,
) -> ConceptRelationNetwork:
    """Build the validated internal relation network over R-06B concepts."""

    return ConceptRelationNetwork(
        foundation=foundation or build_concept_foundation(),
        relations=core_concept_relations(),
    )


__all__ = [
    "ConceptPath",
    "ConceptRelation",
    "ConceptRelationNetwork",
    "ConceptRelationType",
    "RelationSource",
    "RelationStatus",
    "build_concept_relation_network",
    "core_concept_relations",
]
