"""CC3 溯源捕获层 — KPA 七维标注引擎.

核心功能:
- 七维标注的创建、更新、查询
- 15 条 Dy3+ 领域专用标注规则 (自动填充/校验/强制)
- 标注完整度评估与缺失维度提示
- C2PA 式签名生成与验证
- W3C PROV Entity-Activity-Agent 映射

七维标注模型:
1. 来源 (Source) — 原始数据来源 (NIST/DOI/实验条件)
2. 生成 (Generation) — 生成者及生成环境
3. 校验 (Validation) — CC1 四层评审结果
4. 决策 (Decision) — 系统决策路径
5. 演化 (Evolution) — 版本历史与变更
6. 传播 (Propagation) — 使用轨迹与引用
7. 关联 (Relation) — 语义关联网络

融合方案:
- W3C PROV: Entity-Activity-Agent 溯源三元组映射
- C2PA: 加密签名断言 (tamper-evident)
- OpenTelemetry GenAI: trace_id/span_id 标准化传递
- DataCite: DOI 持久标识符验证
- OpenAlex: 期刊权威性分级
- JSON Patch RFC 6902: 演化维度版本 diff
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any

from .models import (
    ChangeType,
    CrossLayerDirection,
    DecisionDimension,
    EvolutionDimension,
    GenerationDimension,
    KPAAnnotation,
    PropagationDimension,
    RelationDimension,
    SourceDimension,
    SourceTier,
    TargetType,
    ValidationDimension,
    ValidationVerdict,
)
from .exceptions import (
    AnnotationNotFoundError,
    CC3Error,
    HashMismatchError,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


# ============================================================
# Dy3+ 领域专用标注规则 (15 条)
# ============================================================


class Dy3AnnotationRule:
    """Dy3+ 领域标注规则基类.

    每条规则定义:
    - rule_id: 规则 ID
    - rule_name: 规则名称
    - dimension: 适用维度
    - severity: 严重级别 (error/warning/info)
    - description: 规则描述
    - apply(): 规则应用逻辑 (返回修改后的维度数据或校验结果)
    """

    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        dimension: str,
        severity: str = "warning",
        description: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.dimension = dimension
        self.severity = severity
        self.description = description

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        """应用规则, 返回结果.

        Returns:
            {"passed": bool, "auto_fixed": bool, "message": str, "dimension": str}
        """
        raise NotImplementedError


# --- 来源维度规则 (R-S01 ~ R-S05) ---


class RS01_DOIFormatRule(Dy3AnnotationRule):
    """R-S01: DOI 格式校验.

    Dy3+ 发光材料领域引用的 DOI 必须符合 ISO 26324 格式.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-S01",
            rule_name="DOI格式校验",
            dimension="source",
            severity="warning",
            description="来源维度中的 DOI 必须符合 10.XXXX/XXXX 格式",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        doi = annotation.source.source_metadata.get("doi", "")
        if not doi:
            return {"passed": True, "auto_fixed": False, "message": "无 DOI, 跳过", "dimension": "source"}
        import re
        pattern = re.compile(r"^10\.\d{4,9}/\S+$")
        clean = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip()
                break
        if pattern.match(clean):
            return {"passed": True, "auto_fixed": True, "message": f"DOI 格式正确: {clean}", "dimension": "source"}
        return {"passed": False, "auto_fixed": False, "message": f"DOI 格式不规范: {doi}", "dimension": "source"}


class RS02_SourceTierRule(Dy3AnnotationRule):
    """R-S02: 来源权威等级自动分级.

    根据 source_type 和期刊名称自动判定 SourceTier.
    """

    TIER1_JOURNALS = {
        "nature", "science", "physical review letters", "prl",
        "jacs", "advanced materials", "acs nano", "nature materials",
        "nature communications", "chemical reviews", "angewandte chemie",
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-S02",
            rule_name="来源权威等级自动分级",
            dimension="source",
            severity="info",
            description="根据来源类型和期刊名称自动判定 SourceTier",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        src = annotation.source
        if not src.source_type:
            return {"passed": True, "auto_fixed": False, "message": "无 source_type", "dimension": "source"}

        journal = src.source_metadata.get("journal", "").lower()
        if src.source_type == "journal" and journal:
            if any(t1 in journal for t1 in self.TIER1_JOURNALS):
                if src.trust_tier != SourceTier.TIER_1:
                    src.trust_tier = SourceTier.TIER_1
                    return {"passed": True, "auto_fixed": True, "message": f"自动升级为 TIER_1 (期刊: {journal})", "dimension": "source"}
            else:
                if src.trust_tier == SourceTier.TIER_3:
                    src.trust_tier = SourceTier.TIER_2
                    return {"passed": True, "auto_fixed": True, "message": f"自动调整为 TIER_2 (同行评审期刊)", "dimension": "source"}
        elif src.source_type == "textbook":
            if src.trust_tier not in (SourceTier.TIER_2, SourceTier.TIER_3):
                src.trust_tier = SourceTier.TIER_2
                return {"passed": True, "auto_fixed": True, "message": "教材来源默认 TIER_2", "dimension": "source"}
        elif src.source_type == "experiment":
            if src.trust_tier == SourceTier.TIER_3:
                src.trust_tier = SourceTier.TIER_2
                return {"passed": True, "auto_fixed": True, "message": "实验数据默认 TIER_2", "dimension": "source"}
        return {"passed": True, "auto_fixed": False, "message": "等级合理", "dimension": "source"}


class RS03_Dy3WavelengthRule(Dy3AnnotationRule):
    """R-S03: Dy3+ 特征发射波长校验.

    Dy3+ 离子的特征发射波长应在已知范围内:
    - 蓝光: ~480 nm (⁴F₉/₂ → ⁶H₁₅/₂)
    - 黄光: ~574 nm (⁴F₉/₂ → ⁶H₁₃/₂)
    - 红光: ~660 nm (⁴F₉/₂ → ⁶H₁₁/₂)
    """

    KNOWN_WAVELENGTHS = [480, 574, 660]

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-S03",
            rule_name="Dy3+特征发射波长校验",
            dimension="source",
            severity="error",
            description="Dy3+ 发射波长必须在已知特征范围内 (480/574/660 nm ±10nm)",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        wavelength = annotation.source.source_metadata.get("emission_wavelength_nm")
        if wavelength is None:
            return {"passed": True, "auto_fixed": False, "message": "无波长信息", "dimension": "source"}
        try:
            wl = float(wavelength)
        except (ValueError, TypeError):
            return {"passed": False, "auto_fixed": False, "message": f"波长值非数值: {wavelength}", "dimension": "source"}

        for known in self.KNOWN_WAVELENGTHS:
            if abs(wl - known) <= 10:
                label = {480: "蓝光(⁴F₉/₂→⁶H₁₅/₂)", 574: "黄光(⁴F₉/₂→⁶H₁₃/₂)", 660: "红光(⁴F₉/₂→⁶H₁₁/₂)"}[known]
                return {"passed": True, "auto_fixed": False, "message": f"波长 {wl}nm 匹配 Dy3+ {label}", "dimension": "source"}
        return {"passed": False, "auto_fixed": False, "message": f"波长 {wl}nm 不在 Dy3+ 已知特征范围", "dimension": "source"}


class RS04_RetrievalTimestampRule(Dy3AnnotationRule):
    """R-S04: 检索时间戳自动填充."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-S04",
            rule_name="检索时间戳自动填充",
            dimension="source",
            severity="info",
            description="来源维度缺少检索时间戳时自动填充当前时间",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.source.retrieval_timestamp == 0.0 and annotation.source.primary_source:
            annotation.source.retrieval_timestamp = time.time()
            return {"passed": True, "auto_fixed": True, "message": "自动填充检索时间戳", "dimension": "source"}
        return {"passed": True, "auto_fixed": False, "message": "时间戳已存在", "dimension": "source"}


class RS05_SecondarySourceRule(Dy3AnnotationRule):
    """R-S05: 次要来源推荐.

    Dy3+ 知识点应有至少一个次要来源用于交叉验证.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-S05",
            rule_name="次要来源推荐",
            dimension="source",
            severity="warning",
            description="Dy3+ 知识点建议有至少一个次要来源用于交叉验证",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.source.primary_source and not annotation.source.secondary_sources:
            return {"passed": False, "auto_fixed": False, "message": "仅有主要来源, 建议添加次要来源交叉验证", "dimension": "source"}
        return {"passed": True, "auto_fixed": False, "message": "次要来源充足", "dimension": "source"}


# --- 生成维度规则 (R-G01 ~ R-G03) ---


class RG01_TraceIDRule(Dy3AnnotationRule):
    """R-G01: OpenTelemetry trace_id 自动填充."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-G01",
            rule_name="TraceID自动填充",
            dimension="generation",
            severity="info",
            description="生成维度缺少 trace_id 时自动生成 OpenTelemetry 格式 ID",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if not annotation.generation.trace_id and annotation.generation.agent_id:
            annotation.generation.trace_id = uuid.uuid4().hex[:32]
            annotation.generation.span_id = uuid.uuid4().hex[:16]
            return {"passed": True, "auto_fixed": True, "message": "自动生成 trace_id 和 span_id", "dimension": "generation"}
        return {"passed": True, "auto_fixed": False, "message": "trace_id 已存在", "dimension": "generation"}


class RG02_CodeHashRule(Dy3AnnotationRule):
    """R-G02: 代码哈希校验."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-G02",
            rule_name="代码哈希校验",
            dimension="generation",
            severity="warning",
            description="生成维度应有 code_hash 以保证可复现性",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.generation.agent_id and not annotation.generation.code_hash:
            return {"passed": False, "auto_fixed": False, "message": "缺少 code_hash, 影响可复现性", "dimension": "generation"}
        return {"passed": True, "auto_fixed": False, "message": "code_hash 已记录", "dimension": "generation"}


class RG03_EnvironmentHashRule(Dy3AnnotationRule):
    """R-G03: 环境哈希自动计算."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-G03",
            rule_name="环境哈希校验",
            dimension="generation",
            severity="warning",
            description="生成维度应有 environment_hash 记录运行环境",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.generation.agent_id and not annotation.generation.environment_hash:
            env_data = json.dumps(annotation.generation.model_cfg, sort_keys=True)
            annotation.generation.environment_hash = hashlib.sha256(env_data.encode()).hexdigest()[:16]
            return {"passed": True, "auto_fixed": True, "message": "自动计算环境哈希", "dimension": "generation"}
        return {"passed": True, "auto_fixed": False, "message": "环境哈希已存在", "dimension": "generation"}


# --- 校验维度规则 (R-V01 ~ R-V02) ---


class RV01_CC1LinkageRule(Dy3AnnotationRule):
    """R-V01: CC1 评审关联校验."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-V01",
            rule_name="CC1评审关联校验",
            dimension="validation",
            severity="warning",
            description="校验维度应有 CC1 评审 ID 关联",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.target_type in (TargetType.KNOWLEDGE_POINT, TargetType.CONTENT, TargetType.REVIEW_REPORT):
            if not annotation.validation.cc1_review_id:
                return {"passed": False, "auto_fixed": False, "message": "知识点/内容缺少 CC1 评审关联", "dimension": "validation"}
        return {"passed": True, "auto_fixed": False, "message": "CC1 关联完整", "dimension": "validation"}


class RV02_FourLayerScoreRule(Dy3AnnotationRule):
    """R-V02: 四层评分完整性校验."""

    REQUIRED_LAYERS = ["factual", "logical", "numerical", "provenance"]

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-V02",
            rule_name="四层评分完整性",
            dimension="validation",
            severity="error",
            description="校验维度的四层评分应包含 factual/logical/numerical/provenance 四项",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        scores = annotation.validation.four_layer_scores
        if not scores and annotation.validation.cc1_review_id:
            return {"passed": False, "auto_fixed": False, "message": "有 CC1 评审 ID 但缺少四层评分", "dimension": "validation"}
        missing = [layer for layer in self.REQUIRED_LAYERS if layer not in scores]
        if missing and scores:
            return {"passed": False, "auto_fixed": False, "message": f"四层评分缺失: {missing}", "dimension": "validation"}
        return {"passed": True, "auto_fixed": False, "message": "四层评分完整", "dimension": "validation"}


# --- 演化维度规则 (R-E01 ~ R-E02) ---


class RE01_VersionChainRule(Dy3AnnotationRule):
    """R-E01: 版本链完整性."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-E01",
            rule_name="版本链完整性",
            dimension="evolution",
            severity="info",
            description="演化维度的版本链应记录创建事件",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.evolution.change_type == ChangeType.CREATED and not annotation.evolution.version_chain:
            annotation.evolution.version_chain.append({
                "version": annotation.evolution.version,
                "timestamp": annotation.created_at,
                "change_type": "created",
                "actor": annotation.annotator_agent,
            })
            return {"passed": True, "auto_fixed": True, "message": "自动添加创建事件到版本链", "dimension": "evolution"}
        return {"passed": True, "auto_fixed": False, "message": "版本链已存在", "dimension": "evolution"}


class RE02_JSONPatchRule(Dy3AnnotationRule):
    """R-E02: JSON Patch 格式校验."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-E02",
            rule_name="JSON Patch格式校验",
            dimension="evolution",
            severity="warning",
            description="演化维度的 diff_snapshot 应符合 RFC 6902 JSON Patch 格式",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        for patch in annotation.evolution.diff_snapshot:
            if not isinstance(patch, dict):
                return {"passed": False, "auto_fixed": False, "message": f"diff_snapshot 项非 dict: {type(patch)}", "dimension": "evolution"}
            if "op" not in patch or "path" not in patch:
                return {"passed": False, "auto_fixed": False, "message": f"JSON Patch 缺少 op/path: {patch}", "dimension": "evolution"}
            if patch["op"] not in ("add", "remove", "replace", "move", "copy", "test"):
                return {"passed": False, "auto_fixed": False, "message": f"非法 op: {patch['op']}", "dimension": "evolution"}
        return {"passed": True, "auto_fixed": False, "message": "JSON Patch 格式正确", "dimension": "evolution"}


# --- 关联维度规则 (R-R01 ~ R-R02) ---


class RR01_PrerequisiteRule(Dy3AnnotationRule):
    """R-R01: 前置知识推荐.

    Dy3+ 知识点应有前置知识关联以构建学习路径.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-R01",
            rule_name="前置知识推荐",
            dimension="relation",
            severity="info",
            description="知识点建议有前置知识关联",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.target_type == TargetType.KNOWLEDGE_POINT:
            if not annotation.relation.prerequisites:
                return {"passed": False, "auto_fixed": False, "message": "知识点缺少前置知识关联", "dimension": "relation"}
        return {"passed": True, "auto_fixed": False, "message": "前置知识关联完整", "dimension": "relation"}


class RR02_Dy3DomainRelationRule(Dy3AnnotationRule):
    """R-R02: Dy3+ 领域关联自动识别.

    自动识别 Dy3+ 领域常见关联模式.
    """

    DY3_KEYWORDS = {
        "judd-ofelt": "Judd-Ofelt理论参数",
        "浓度猝灭": "浓度猝灭效应",
        "yag": "YAG基质",
        "cct": "相关色温(CCT)",
        "量子效率": "量子效率(QY)",
        "cie": "CIE色度坐标",
        "cristallinity": "结晶度",
        "fwhm": "半峰全宽(FWHM)",
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-R02",
            rule_name="Dy3+领域关联自动识别",
            dimension="relation",
            severity="info",
            description="自动识别 Dy3+ 领域常见关联模式",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        target_meta = json.dumps(annotation.target_metadata, ensure_ascii=False).lower()
        found = []
        for keyword, label in self.DY3_KEYWORDS.items():
            if keyword.lower() in target_meta and not any(
                r.get("label") == label for r in annotation.relation.same_domain_relations
            ):
                annotation.relation.same_domain_relations.append({
                    "target_id": f"auto-{keyword.replace(' ', '-')}",
                    "relation_type": "same_domain",
                    "strength": 0.5,
                    "label": label,
                })
                found.append(label)
        if found:
            return {"passed": True, "auto_fixed": True, "message": f"自动添加关联: {', '.join(found)}", "dimension": "relation"}
        return {"passed": True, "auto_fixed": False, "message": "无匹配关联", "dimension": "relation"}


# --- 传播维度规则 (R-P01) ---


class RP01_PropagationInitRule(Dy3AnnotationRule):
    """R-P01: 传播维度初始化."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="R-P01",
            rule_name="传播维度初始化",
            dimension="propagation",
            severity="info",
            description="新创建标注的传播维度应初始化基本字段",
        )

    def apply(self, annotation: KPAAnnotation) -> dict[str, Any]:
        if annotation.propagation.last_accessed_at == 0.0:
            annotation.propagation.last_accessed_at = annotation.created_at
            return {"passed": True, "auto_fixed": True, "message": "初始化 last_accessed_at", "dimension": "propagation"}
        return {"passed": True, "auto_fixed": False, "message": "传播维度已初始化", "dimension": "propagation"}


# 规则注册表
DY3_ANNOTATION_RULES: list[Dy3AnnotationRule] = [
    RS01_DOIFormatRule(),
    RS02_SourceTierRule(),
    RS03_Dy3WavelengthRule(),
    RS04_RetrievalTimestampRule(),
    RS05_SecondarySourceRule(),
    RG01_TraceIDRule(),
    RG02_CodeHashRule(),
    RG03_EnvironmentHashRule(),
    RV01_CC1LinkageRule(),
    RV02_FourLayerScoreRule(),
    RE01_VersionChainRule(),
    RE02_JSONPatchRule(),
    RR01_PrerequisiteRule(),
    RR02_Dy3DomainRelationRule(),
    RP01_PropagationInitRule(),
]


# ============================================================
# KPA 标注引擎
# ============================================================


class KPAEngine:
    """KPA 七维标注引擎.

    提供标注的完整生命周期管理:
    - 创建标注 (支持七维部分填充)
    - 更新标注维度 (渐进完善)
    - 应用 Dy3+ 领域规则 (自动填充/校验)
    - 完整度评估与缺失维度提示
    - C2PA 式签名生成与验证
    - W3C PROV 三元组映射
    - 标注查询与统计

    使用示例::

        engine = KPAEngine()
        annotation = engine.create_annotation(
            target_type=TargetType.KNOWLEDGE_POINT,
            target_id="kp-dy3-yag-4f",
            target_metadata={"title": "Dy3+在YAG中的4f-4f跃迁"},
            source=SourceDimension(
                primary_source="10.1016/j.jlumin.2019.116789",
                source_type="journal",
            ),
        )
        report = engine.apply_rules(annotation)
        print(report["summary"])
    """

    def __init__(self, signing_key: str = "") -> None:
        """初始化 KPA 引擎.

        Args:
            signing_key: C2PA 签名密钥 (为空则不生成签名)
        """
        self._annotations: dict[str, KPAAnnotation] = {}
        self._target_index: dict[str, list[str]] = {}  # target_id -> [annotation_id]
        self._signing_key = signing_key.encode() if signing_key else b""
        self._lock = __import__("threading").RLock()
        self._rules: list[Dy3AnnotationRule] = list(DY3_ANNOTATION_RULES)

    # ==========================================================
    # 标注创建
    # ==========================================================

    def create_annotation(
        self,
        target_type: TargetType = TargetType.KNOWLEDGE_POINT,
        target_id: str = "",
        target_metadata: dict[str, Any] | None = None,
        source: SourceDimension | None = None,
        generation: GenerationDimension | None = None,
        validation: ValidationDimension | None = None,
        decision: DecisionDimension | None = None,
        evolution: EvolutionDimension | None = None,
        propagation: PropagationDimension | None = None,
        relation: RelationDimension | None = None,
        annotator_agent: str = "cc3-provenance-agent",
    ) -> KPAAnnotation:
        """创建新的 KPA 标注.

        支持七维部分填充, 未提供的维度使用默认值.
        创建后自动应用 Dy3+ 领域规则.

        Args:
            target_type: 标注对象类型
            target_id: 标注对象 ID
            target_metadata: 标注对象元数据
            source ~ relation: 七维标注数据 (可选)
            annotator_agent: 标注 Agent ID

        Returns:
            新创建的 KPAAnnotation
        """
        with self._lock:
            annotation = KPAAnnotation(
                target_type=target_type,
                target_id=target_id,
                target_metadata=target_metadata or {},
                source=source or SourceDimension(),
                generation=generation or GenerationDimension(),
                validation=validation or ValidationDimension(),
                decision=decision or DecisionDimension(),
                evolution=evolution or EvolutionDimension(),
                propagation=propagation or PropagationDimension(),
                relation=relation or RelationDimension(),
                annotator_agent=annotator_agent,
            )

            # 应用 Dy3+ 规则
            self._apply_rules_internal(annotation)

            # 重新计算哈希 (规则可能修改了数据, 必须在签名前完成)
            annotation.immutable_hash = annotation.compute_hash()
            annotation.updated_at = time.time()

            # 生成 C2PA 签名 (基于最终哈希, 确保验证一致)
            if self._signing_key:
                annotation.signature = self._sign(annotation)

            self._annotations[annotation.annotation_id] = annotation
            if target_id:
                self._target_index.setdefault(target_id, []).append(annotation.annotation_id)

            logger.info(
                "创建 KPA 标注: id=%s, target=%s/%s, completeness=%.2f",
                annotation.annotation_id,
                target_type.value,
                target_id,
                annotation.completeness_score(),
            )
            return annotation

    # ==========================================================
    # 标注更新
    # ==========================================================

    def update_dimension(
        self,
        annotation_id: str,
        dimension: str,
        data: dict[str, Any],
    ) -> KPAAnnotation:
        """更新标注的指定维度.

        支持渐进完善: 每次更新一个或多个维度字段.
        更新后自动重新应用规则并重算哈希.

        Args:
            annotation_id: 标注 ID
            dimension: 维度名称 (source/generation/validation/decision/evolution/propagation/relation)
            data: 要更新的字段字典

        Returns:
            更新后的 KPAAnnotation

        Raises:
            AnnotationNotFoundError: 标注不存在
            SchemaValidationError: 维度名称非法
        """
        valid_dims = {"source", "generation", "validation", "decision", "evolution", "propagation", "relation"}
        if dimension not in valid_dims:
            raise SchemaValidationError(dimension, list(valid_dims), f"非法维度名: {dimension}")

        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)

            dim_obj = getattr(annotation, dimension)
            for key, value in data.items():
                if hasattr(dim_obj, key):
                    setattr(dim_obj, key, value)

            # 记录演化
            annotation.evolution.version_chain.append({
                "version": annotation.evolution.version,
                "timestamp": time.time(),
                "change_type": "enhanced",
                "actor": annotation.annotator_agent,
                "dimension": dimension,
            })

            # 重新应用规则
            self._apply_rules_internal(annotation)

            # 重算哈希
            annotation.immutable_hash = annotation.compute_hash()
            annotation.updated_at = time.time()

            logger.debug("更新标注维度: id=%s, dim=%s, fields=%s", annotation_id, dimension, list(data.keys()))
            return annotation

    def update_validation(
        self,
        annotation_id: str,
        cc1_review_id: str,
        four_layer_scores: dict[str, float],
        verdict: ValidationVerdict = ValidationVerdict.PASS,
        issues: list[dict[str, Any]] | None = None,
        self_correction_count: int = 0,
    ) -> KPAAnnotation:
        """便捷方法: 更新校验维度 (CC1 评审结果).

        Args:
            annotation_id: 标注 ID
            cc1_review_id: CC1 评审报告 ID
            four_layer_scores: 四层评分
            verdict: 评审结论
            issues: 问题列表
            self_correction_count: 自纠回路迭代次数

        Returns:
            更新后的 KPAAnnotation
        """
        return self.update_dimension(
            annotation_id,
            "validation",
            {
                "cc1_review_id": cc1_review_id,
                "four_layer_scores": four_layer_scores,
                "verdict": verdict,
                "validation_issues": issues or [],
                "self_correction_count": self_correction_count,
                "validated_at": time.time(),
            },
        )

    def update_decision(
        self,
        annotation_id: str,
        meta_decider_result: str = "",
        paradigm_selected: str = "",
        cc2_approval_id: str = "",
        cc2_approval_level: str = "",
        debate_id: str = "",
        decision_path: list[str] | None = None,
    ) -> KPAAnnotation:
        """便捷方法: 更新决策维度 (CC2 审批/辩论结果).

        Args:
            annotation_id: 标注 ID
            meta_decider_result: Meta-Decider 决策结果
            paradigm_selected: 选择的讲解范式
            cc2_approval_id: CC2 审批记录 ID
            cc2_approval_level: CC2 协同层级
            debate_id: 辩论 ID
            decision_path: 决策路径

        Returns:
            更新后的 KPAAnnotation
        """
        return self.update_dimension(
            annotation_id,
            "decision",
            {
                "meta_decider_result": meta_decider_result,
                "paradigm_selected": paradigm_selected,
                "cc2_approval_id": cc2_approval_id,
                "cc2_approval_level": cc2_approval_level,
                "debate_id": debate_id,
                "decision_path": decision_path or [],
                "decision_timestamp": time.time(),
            },
        )

    def record_propagation(
        self,
        annotation_id: str,
        session_id: str = "",
        agent_id: str = "",
        learner_id: str = "",
        interaction_type: str = "",
    ) -> KPAAnnotation:
        """记录传播轨迹 (使用/引用).

        Args:
            annotation_id: 标注 ID
            session_id: 会话 ID
            agent_id: 使用 Agent ID
            learner_id: 学习者 ID
            interaction_type: 交互类型

        Returns:
            更新后的 KPAAnnotation
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)

            now = time.time()
            if session_id and session_id not in annotation.propagation.session_references:
                annotation.propagation.session_references.append(session_id)
            if agent_id:
                annotation.propagation.agent_usages.append({
                    "agent_id": agent_id,
                    "timestamp": now,
                    "context": interaction_type,
                })
            if learner_id:
                annotation.propagation.learner_consumptions.append({
                    "learner_id": learner_id,
                    "timestamp": now,
                    "interaction_type": interaction_type,
                })
            annotation.propagation.citation_count += 1
            annotation.propagation.last_accessed_at = now
            annotation.immutable_hash = annotation.compute_hash()
            annotation.updated_at = now
            return annotation

    # ==========================================================
    # Dy3+ 规则应用
    # ==========================================================

    def apply_rules(self, annotation_id: str) -> dict[str, Any]:
        """对指定标注应用全部 Dy3+ 规则.

        Args:
            annotation_id: 标注 ID

        Returns:
            规则应用报告::

                {
                    "annotation_id": str,
                    "total_rules": int,
                    "passed": int,
                    "failed": int,
                    "auto_fixed": int,
                    "errors": [...],
                    "warnings": [...],
                    "info": [...],
                    "summary": str,
                }
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)
            return self._apply_rules_internal(annotation)

    def _apply_rules_internal(self, annotation: KPAAnnotation) -> dict[str, Any]:
        """内部规则应用 (不加锁)."""
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []
        auto_fixed_count = 0
        passed_count = 0
        failed_count = 0

        for rule in self._rules:
            try:
                result = rule.apply(annotation)
                result["rule_id"] = rule.rule_id
                result["rule_name"] = rule.rule_name
                results.append(result)

                if result["auto_fixed"]:
                    auto_fixed_count += 1
                if result["passed"]:
                    passed_count += 1
                else:
                    failed_count += 1
                    if rule.severity == "error":
                        errors.append(result)
                    elif rule.severity == "warning":
                        warnings.append(result)
                    else:
                        infos.append(result)
            except Exception as exc:
                logger.warning("规则 %s 应用异常: %s", rule.rule_id, exc)
                errors.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "passed": False,
                    "auto_fixed": False,
                    "message": f"规则异常: {exc}",
                    "dimension": rule.dimension,
                })
                failed_count += 1

        summary = (
            f"规则应用完成: {passed_count}/{len(self._rules)} 通过, "
            f"{failed_count} 失败, {auto_fixed_count} 自动修复"
        )

        return {
            "annotation_id": annotation.annotation_id,
            "total_rules": len(self._rules),
            "passed": passed_count,
            "failed": failed_count,
            "auto_fixed": auto_fixed_count,
            "errors": errors,
            "warnings": warnings,
            "info": infos,
            "all_results": results,
            "summary": summary,
        }

    # ==========================================================
    # 完整度评估
    # ==========================================================

    def evaluate_completeness(self, annotation_id: str) -> dict[str, Any]:
        """评估标注的七维完整度.

        Args:
            annotation_id: 标注 ID

        Returns:
            完整度报告::

                {
                    "annotation_id": str,
                    "overall_score": float,
                    "dimension_scores": {...},
                    "filled_dimensions": [...],
                    "missing_dimensions": [...],
                    "recommendations": [...],
                }
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)

            dim_scores = {
                "source": annotation.source.completeness(),
                "generation": annotation.generation.completeness(),
                "validation": annotation.validation.completeness(),
                "decision": annotation.decision.completeness(),
                "evolution": annotation.evolution.completeness(),
                "propagation": annotation.propagation.completeness(),
                "relation": annotation.relation.completeness(),
            }
            filled = annotation.filled_dimensions()
            missing = annotation.missing_dimensions()
            recommendations: list[str] = []

            if "source" in missing:
                recommendations.append("填充来源维度: 添加 primary_source (DOI/URL/NIST标准号)")
            if "generation" in missing:
                recommendations.append("填充生成维度: 添加 agent_id 和 agent_version")
            if "validation" in missing:
                recommendations.append("填充校验维度: 关联 CC1 评审报告 ID")
            if "decision" in missing:
                recommendations.append("填充决策维度: 记录 Meta-Decider 决策结果")
            if "relation" in missing:
                recommendations.append("填充关联维度: 添加前置/后继知识 ID")

            return {
                "annotation_id": annotation_id,
                "overall_score": annotation.completeness_score(),
                "dimension_scores": {k: round(v, 4) for k, v in dim_scores.items()},
                "filled_dimensions": filled,
                "missing_dimensions": missing,
                "recommendations": recommendations,
            }

    # ==========================================================
    # C2PA 签名
    # ==========================================================

    def _sign(self, annotation: KPAAnnotation) -> str:
        """生成 C2PA 式 HMAC-SHA256 签名."""
        payload = annotation.immutable_hash or annotation.compute_hash()
        return hmac.new(self._signing_key, payload.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, annotation_id: str) -> bool:
        """验证标注的 C2PA 签名.

        Args:
            annotation_id: 标注 ID

        Returns:
            签名是否有效
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)
            if not annotation.signature or not self._signing_key:
                return False
            expected = self._sign(annotation)
            return hmac.compare_digest(expected, annotation.signature)

    def verify_hash(self, annotation_id: str) -> bool:
        """验证标注的不可变哈希.

        Args:
            annotation_id: 标注 ID

        Returns:
            哈希是否匹配

        Raises:
            HashMismatchError: 哈希不匹配 (可能被篡改)
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)
            if not annotation.verify_hash():
                raise HashMismatchError(
                    expected_hash=annotation.immutable_hash,
                    actual_hash=annotation.compute_hash(),
                    record_id=annotation_id,
                )
            return True

    # ==========================================================
    # W3C PROV 映射
    # ==========================================================

    to_prov: dict[str, str]

    def to_prov_model(self, annotation_id: str) -> dict[str, Any]:
        """将标注映射为 W3C PROV Entity-Activity-Agent 三元组.

        PROV 模型:
        - Entity: 标注对象 (target_id)
        - Activity: 生成/校验/决策过程
        - Agent: 生成 Agent / 标注 Agent / CC1 评审器

        Args:
            annotation_id: 标注 ID

        Returns:
            PROV 格式字典
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)

            return {
                "entities": [
                    {
                        "id": annotation.target_id or annotation.annotation_id,
                        "type": annotation.target_type.value,
                        "metadata": annotation.target_metadata,
                    }
                ],
                "activities": [
                    {
                        "id": f"act-gen-{annotation.annotation_id}",
                        "type": "generation",
                        "agent": annotation.generation.agent_id,
                        "timestamp": annotation.generation.generation_timestamp,
                    },
                    {
                        "id": f"act-val-{annotation.annotation_id}",
                        "type": "validation",
                        "agent": "cc1-reviewer",
                        "timestamp": annotation.validation.validated_at,
                    } if annotation.validation.cc1_review_id else None,
                    {
                        "id": f"act-ann-{annotation.annotation_id}",
                        "type": "annotation",
                        "agent": annotation.annotator_agent,
                        "timestamp": annotation.created_at,
                    },
                ],
                "agents": [
                    {
                        "id": annotation.generation.agent_id or "unknown",
                        "type": "software-agent",
                        "version": annotation.generation.agent_version,
                    },
                    {
                        "id": annotation.annotator_agent,
                        "type": "software-agent",
                    },
                ],
                "relations": [
                    {"type": "wasGeneratedBy", "entity": annotation.target_id, "activity": f"act-gen-{annotation.annotation_id}"},
                    {"type": "wasAttributedTo", "entity": annotation.target_id, "agent": annotation.generation.agent_id},
                    {"type": "wasDerivedFrom", "entity": annotation.target_id, "source": annotation.source.primary_source}
                    if annotation.source.primary_source else None,
                ],
                "provenance_hash": annotation.immutable_hash,
            }

    # ==========================================================
    # 查询
    # ==========================================================

    def get_annotation(self, annotation_id: str) -> KPAAnnotation:
        """获取标注.

        Raises:
            AnnotationNotFoundError: 标注不存在
        """
        with self._lock:
            annotation = self._annotations.get(annotation_id)
            if annotation is None:
                raise AnnotationNotFoundError(annotation_id)
            return annotation

    def get_by_target(self, target_id: str) -> list[KPAAnnotation]:
        """按目标 ID 查询标注列表."""
        with self._lock:
            ids = self._target_index.get(target_id, [])
            return [self._annotations[aid] for aid in ids if aid in self._annotations]

    def list_annotations(
        self,
        target_type: TargetType | None = None,
        min_completeness: float = 0.0,
        limit: int = 100,
    ) -> list[KPAAnnotation]:
        """列出标注.

        Args:
            target_type: 按类型筛选 (None=全部)
            min_completeness: 最低完整度
            limit: 最多返回数

        Returns:
            标注列表
        """
        with self._lock:
            results = []
            for annotation in self._annotations.values():
                if target_type is not None and annotation.target_type != target_type:
                    continue
                if annotation.completeness_score() < min_completeness:
                    continue
                results.append(annotation)
                if len(results) >= limit:
                    break
            return results

    # ==========================================================
    # 统计
    # ==========================================================

    def statistics(self) -> dict[str, Any]:
        """获取标注统计信息."""
        with self._lock:
            total = len(self._annotations)
            if total == 0:
                return {"total": 0}

            by_type: dict[str, int] = {}
            completeness_scores: list[float] = []
            filled_dims_count: dict[str, int] = {}

            for annotation in self._annotations.values():
                t = annotation.target_type.value
                by_type[t] = by_type.get(t, 0) + 1
                completeness_scores.append(annotation.completeness_score())
                for dim in annotation.filled_dimensions():
                    filled_dims_count[dim] = filled_dims_count.get(dim, 0) + 1

            avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

            return {
                "total": total,
                "by_type": by_type,
                "avg_completeness": round(avg_completeness, 4),
                "min_completeness": round(min(completeness_scores), 4) if completeness_scores else 0.0,
                "max_completeness": round(max(completeness_scores), 4) if completeness_scores else 0.0,
                "filled_dimensions": filled_dims_count,
                "total_targets": len(self._target_index),
            }

    # ==========================================================
    # 清空 (测试用)
    # ==========================================================

    def clear(self) -> None:
        """清空所有标注."""
        with self._lock:
            self._annotations.clear()
            self._target_index.clear()


__all__ = [
    "Dy3AnnotationRule",
    "DY3_ANNOTATION_RULES",
    "KPAEngine",
    # 规则类
    "RS01_DOIFormatRule",
    "RS02_SourceTierRule",
    "RS03_Dy3WavelengthRule",
    "RS04_RetrievalTimestampRule",
    "RS05_SecondarySourceRule",
    "RG01_TraceIDRule",
    "RG02_CodeHashRule",
    "RG03_EnvironmentHashRule",
    "RV01_CC1LinkageRule",
    "RV02_FourLayerScoreRule",
    "RE01_VersionChainRule",
    "RE02_JSONPatchRule",
    "RR01_PrerequisiteRule",
    "RR02_Dy3DomainRelationRule",
    "RP01_PropagationInitRule",
]
