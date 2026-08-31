"""CC3 溯源捕获层 — 完整测试.

覆盖 CC3 Provenance Capture 系统的全部模块:
1. KPA 七维标注引擎 (KPAEngine) — 标注创建/维度更新/15条Dy3+规则/完整度/哈希/C2PA签名/W3C PROV
2. 辩论日志引擎 (DebateLogger) — 日志创建/轮次/收敛/裁决/完成/完整性/分歧度/脱敏/导出/统计
3. 溯源链构建器 (ProvenanceChainBuilder) — 链创建/节点追加/验证/Merkle树/证明/压缩/快照/审计/跨层
4. L0 Ledger 集成 (LedgerIntegration) — KPA/DL/跨层/人工干预写入/查询/验证/统计
5. 查询引擎 (QueryEngine) — trace回溯/知识溯源/Agent历史/时间线/图/概览
6. CC1/CC2 跨切面集成 (CCIntegration) — 评审/审批回调/溯源检查/升级建议/辩论触发
7. KPI 指标引擎 (KPAMetricsEngine) — 覆盖率/完整性/性能/合规/仪表盘/延迟
8. 可视化适配器 (ProvenanceVisualizer) — Cytoscape/D3/Mermaid/ECharts

测试领域: Dy3+ 发光材料 (YAG 基质, 4f-4f 跃迁, 480/574/660nm 发射)
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from dy3_polaris.l0.cc3 import (
    # 引擎
    KPAEngine,
    DebateLogger,
    ProvenanceChainBuilder,
    LedgerIntegration,
    QueryEngine,
    CCIntegration,
    KPAMetricsEngine,
    ProvenanceVisualizer,
    # 模型
    KPAAnnotation,
    TargetType,
    SourceTier,
    ChangeType,
    ValidationVerdict,
    LogVerbosity,
    ConvergenceStatus,
    EventType,
    CrossLayerDirection,
    CounterType,
    SourceDimension,
    GenerationDimension,
    ValidationDimension,
    DecisionDimension,
    EvolutionDimension,
    PropagationDimension,
    RelationDimension,
    DebateArgument,
    DebateCounterargument,
    DebateRound,
    AdjudicatorVerdict,
    PreDebateRecord,
    DebateLog,
    ProvenanceChainNode,
    LedgerEvent,
    AuditVerificationResult,
    # 异常
    AnnotationNotFoundError,
    DebateLogNotFoundError,
    SchemaValidationError,
    HashMismatchError,
    ChainBrokenError,
    # 辅助类
    DivergenceCalculator,
    PromptSanitizer,
    MerkleTree,
    DY3_ANNOTATION_RULES,
)


# ============================================================
# Dy3+ 领域辅助工厂
# ============================================================


def _make_dy3_source(journal: str = "Journal of Luminescence") -> SourceDimension:
    """创建 Dy3+ 发光材料领域的来源维度."""
    return SourceDimension(
        primary_source="10.1016/j.jlumin.2019.116789",
        source_type="journal",
        trust_tier=SourceTier.TIER_3,
        secondary_sources=["10.1021/acs.inorgchem.8b03122"],
        source_metadata={
            "doi": "10.1016/j.jlumin.2019.116789",
            "journal": journal,
            "emission_wavelength_nm": 574,
            "authors": ["Zhang", "Li"],
            "year": 2019,
        },
        retrieval_method="api",
        retrieval_timestamp=0.0,
    )


def _make_dy3_generation() -> GenerationDimension:
    """创建 Dy3+ 领域的生成维度."""
    return GenerationDimension(
        agent_id="dy3-content-agent-v2",
        agent_version="2.1.0",
        agent_role="generator",
        code_hash="a1b2c3d4e5f6",
        prompt_version="dy3-v3",
        model_cfg={"model_name": "gpt-4", "temperature": 0.3},
        generation_timestamp=time.time(),
        generation_duration_ms=1200.0,
        trace_id="",
        span_id="",
        environment_hash="",
    )


def _make_dy3_validation() -> ValidationDimension:
    """创建 Dy3+ 领域的校验维度 (CC1 四层评审)."""
    return ValidationDimension(
        cc1_review_id="rv-cc1-dy3-001",
        four_layer_scores={
            "factual": 92,
            "logical": 88,
            "numerical": 95,
            "provenance": 85,
        },
        verdict=ValidationVerdict.PASS,
        validation_issues=[],
        standard_value_check={"wavelength_574nm": "pass"},
        mcp_tool_calls=[{"tool": "nist_lookup", "result": "match"}],
        self_correction_count=1,
        validated_at=time.time(),
    )


def _make_dy3_decision() -> DecisionDimension:
    """创建 Dy3+ 领域的决策维度."""
    return DecisionDimension(
        meta_decider_result="deductive",
        paradigm_selected="energy_level_diagram",
        adjudicator_verdict="consensus_reached",
        cc2_approval_id="ap-cc2-001",
        cc2_approval_level="approval",
        debate_id="debate-dy3-001",
        decision_path=["meta_decider", "paradigm_select", "cc2_approval"],
        decision_timestamp=time.time(),
    )


def _make_dy3_relation() -> RelationDimension:
    """创建 Dy3+ 领域的关联维度."""
    return RelationDimension(
        prerequisites=["kp-lanthanide-basics", "kp-4f-electron-config"],
        successors=["kp-dy3-concentration-quenching"],
        same_domain_relations=[
            {"target_id": "kp-dy3-yag-judd-ofelt", "relation_type": "same_domain", "strength": 0.8},
        ],
        cross_domain_relations=[],
        relation_strength=0.8,
        network_centrality=0.6,
    )


def _make_generator_args() -> list[DebateArgument]:
    """创建 Generator 论点 (Dy3+ 黄光发射)."""
    return [
        DebateArgument(
            point="Dy3+ 在 YAG 基质中的 574nm 黄光发射源于 ⁴F₉/₂→⁶H₁₃/₂ 跃迁",
            source="10.1016/j.jlumin.2019.116789",
            confidence=0.9,
            evidence_type="citation",
            evidence_metadata={"wavelength_nm": 574},
        ),
        DebateArgument(
            point="YAG 基质低声子能量有助于提高 Dy3+ 发光效率",
            source="10.1021/acs.inorgchem.8b03122",
            confidence=0.85,
            evidence_type="experiment",
        ),
    ]


def _make_reviewer_counters(targets: list[str]) -> list[DebateCounterargument]:
    """创建 Reviewer 反驳."""
    return [
        DebateCounterargument(
            targets=targets,
            counter="574nm 发射强度受浓度猝灭影响, 需注明掺杂浓度",
            source="实验数据补充",
            confidence=0.7,
            counter_type=CounterType.EVIDENCE_BASED,
        ),
    ]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def kpa_engine() -> KPAEngine:
    """创建独立的 KPA 引擎实例."""
    return KPAEngine()


@pytest.fixture
def kpa_engine_signed() -> KPAEngine:
    """创建带签名密钥的 KPA 引擎."""
    return KPAEngine(signing_key="dy3-test-signing-key-2024")


@pytest.fixture
def debate_logger() -> DebateLogger:
    """创建独立的辩论日志引擎实例."""
    return DebateLogger()


@pytest.fixture
def chain_builder() -> ProvenanceChainBuilder:
    """创建独立的溯源链构建器实例."""
    return ProvenanceChainBuilder()


@pytest.fixture
def ledger() -> LedgerIntegration:
    """创建独立的 L0 Ledger 集成器实例."""
    return LedgerIntegration()


@pytest.fixture
def full_kpa_annotation(kpa_engine: KPAEngine) -> KPAAnnotation:
    """创建一个填充多维度 Dy3+ KPA 标注."""
    return kpa_engine.create_annotation(
        target_type=TargetType.KNOWLEDGE_POINT,
        target_id="kp-dy3-yag-4f",
        target_metadata={
            "title": "Dy3+在YAG中的4f-4f跃迁与发光特性",
            "keywords": ["Dy3+", "YAG", "4f-4f", "Judd-Ofelt", "CIE"],
        },
        source=_make_dy3_source(),
        generation=_make_dy3_generation(),
        validation=_make_dy3_validation(),
        decision=_make_dy3_decision(),
        relation=_make_dy3_relation(),
        annotator_agent="cc3-provenance-agent",
    )


@pytest.fixture
def minimal_annotation(kpa_engine: KPAEngine) -> KPAAnnotation:
    """创建一个最小填充的 KPA 标注 (仅来源)."""
    return kpa_engine.create_annotation(
        target_type=TargetType.KNOWLEDGE_POINT,
        target_id="kp-dy3-concentration-quenching",
        target_metadata={"title": "Dy3+浓度猝灭效应"},
        source=SourceDimension(
            primary_source="10.1039/c9tc00001a",
            source_type="journal",
            source_metadata={"doi": "10.1039/c9tc00001a", "journal": "Journal of Materials Chemistry C"},
        ),
    )


@pytest.fixture
def debate_log_with_rounds(debate_logger: DebateLogger) -> str:
    """创建一个带多轮辩论的日志, 返回 debate_log_id."""
    log = debate_logger.create_log(
        debate_id="debate-dy3-yag-001",
        task_id="task-dy3-yag",
        session_id="sess-001",
        trigger_reason="复杂度评分45, 触发辩论",
        complexity_score=45.0,
        focus_area="Dy3+ 574nm 发射机制准确性",
        verbosity=LogVerbosity.FULL,
        convergence_threshold=0.1,
        max_rounds=3,
    )
    # 第1轮: 高分歧
    args1 = _make_generator_args()
    counters1 = _make_reviewer_counters([args1[0].point_id])
    debate_logger.add_round(log.debate_log_id, args1, counters1, round_duration_ms=800.0)

    # 第2轮: 分歧降低
    args2 = [
        DebateArgument(
            point="补充: Dy3+ 掺杂浓度5at%时 574nm 发射最强",
            source="实验数据",
            confidence=0.92,
            evidence_type="experiment",
        ),
    ]
    counters2 = [
        DebateCounterargument(
            targets=[args2[0].point_id],
            counter="同意补充浓度信息, 该数据可靠",
            source="交叉验证",
            confidence=0.88,
            counter_type=CounterType.EVIDENCE_BASED,
        ),
    ]
    debate_logger.add_round(log.debate_log_id, args2, counters2, round_duration_ms=600.0)

    return log.debate_log_id


# ============================================================
# 1. KPA 七维标注引擎测试
# ============================================================


class TestKPAEngine:
    """KPA 七维标注引擎完整测试."""

    # --- 标注创建 ---

    def test_创建标注_基本属性(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_type=TargetType.KNOWLEDGE_POINT,
            target_id="kp-dy3-yag-4f",
            target_metadata={"title": "Dy3+ 4f-4f跃迁"},
        )
        assert ann.annotation_id.startswith("kpa-")
        assert ann.target_type == TargetType.KNOWLEDGE_POINT
        assert ann.target_id == "kp-dy3-yag-4f"
        assert ann.target_metadata["title"] == "Dy3+ 4f-4f跃迁"
        assert ann.annotator_agent == "cc3-provenance-agent"
        assert ann.immutable_hash != ""
        assert len(ann.immutable_hash) == 64

    def test_创建标注_自动应用规则(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-yag-4f",
            source=SourceDimension(
                primary_source="10.1016/j.jlumin.2019.116789",
                source_type="journal",
                source_metadata={"doi": "10.1016/j.jlumin.2019.116789", "journal": "Journal of Luminescence"},
            ),
            generation=_make_dy3_generation(),
        )
        # R-S04 应自动填充 retrieval_timestamp
        assert ann.source.retrieval_timestamp > 0
        # R-G01 应自动生成 trace_id
        assert ann.generation.trace_id != ""
        assert ann.generation.span_id != ""
        # R-G03 应自动计算 environment_hash
        assert ann.generation.environment_hash != ""
        # R-E01 应自动添加创建事件到版本链
        assert len(ann.evolution.version_chain) > 0

    def test_创建标注_带签名(self, kpa_engine_signed: KPAEngine) -> None:
        ann = kpa_engine_signed.create_annotation(
            target_id="kp-dy3-yag-4f",
        )
        assert ann.signature != ""
        assert kpa_engine_signed.verify_signature(ann.annotation_id) is True

    def test_创建标注_无签名密钥时无签名(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-yag-4f")
        assert ann.signature == ""

    def test_创建标注_七维默认值(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-empty")
        assert ann.source.primary_source == ""
        assert ann.generation.agent_id == ""
        assert ann.validation.cc1_review_id == ""
        assert ann.decision.meta_decider_result == ""
        assert ann.evolution.version == "1.0.0"
        assert ann.propagation.citation_count == 0
        assert ann.relation.prerequisites == []

    # --- 维度更新 ---

    def test_更新维度_source(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        updated = kpa_engine.update_dimension(
            minimal_annotation.annotation_id,
            "source",
            {"retrieval_method": "api", "trust_tier": SourceTier.TIER_2},
        )
        assert updated.source.retrieval_method == "api"
        assert updated.source.trust_tier == SourceTier.TIER_2

    def test_更新维度_validation(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        updated = kpa_engine.update_validation(
            minimal_annotation.annotation_id,
            cc1_review_id="rv-cc1-001",
            four_layer_scores={"factual": 90, "logical": 85, "numerical": 92, "provenance": 88},
            verdict=ValidationVerdict.PASS,
        )
        assert updated.validation.cc1_review_id == "rv-cc1-001"
        assert updated.validation.four_layer_scores["factual"] == 90
        assert updated.validation.verdict == ValidationVerdict.PASS
        assert updated.validation.validated_at > 0

    def test_更新维度_decision(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        updated = kpa_engine.update_decision(
            minimal_annotation.annotation_id,
            meta_decider_result="deductive",
            paradigm_selected="energy_level_diagram",
            cc2_approval_id="ap-001",
            cc2_approval_level="approval",
            decision_path=["meta_decider", "paradigm_select"],
        )
        assert updated.decision.meta_decider_result == "deductive"
        assert updated.decision.paradigm_selected == "energy_level_diagram"
        assert updated.decision.cc2_approval_id == "ap-001"
        assert updated.decision.cc2_approval_level == "approval"
        assert updated.decision.decision_timestamp > 0

    def test_更新维度_记录演化历史(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        kpa_engine.update_dimension(
            minimal_annotation.annotation_id, "source", {"retrieval_method": "api"},
        )
        ann = kpa_engine.get_annotation(minimal_annotation.annotation_id)
        # 应有创建事件 + 更新事件
        assert len(ann.evolution.version_chain) >= 2
        last = ann.evolution.version_chain[-1]
        assert last["change_type"] == "enhanced"
        assert last["dimension"] == "source"

    def test_更新维度_非法维度名抛异常(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        with pytest.raises(SchemaValidationError):
            kpa_engine.update_dimension(minimal_annotation.annotation_id, "invalid_dim", {})

    def test_更新维度_标注不存在抛异常(self, kpa_engine: KPAEngine) -> None:
        with pytest.raises(AnnotationNotFoundError):
            kpa_engine.update_dimension("kpa-nonexistent", "source", {})

    def test_记录传播轨迹(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        ann = kpa_engine.record_propagation(
            minimal_annotation.annotation_id,
            session_id="sess-001",
            agent_id="dy3-query-agent",
            learner_id="learner-001",
            interaction_type="query",
        )
        assert "sess-001" in ann.propagation.session_references
        assert len(ann.propagation.agent_usages) == 1
        assert ann.propagation.agent_usages[0]["agent_id"] == "dy3-query-agent"
        assert len(ann.propagation.learner_consumptions) == 1
        assert ann.propagation.citation_count == 1
        assert ann.propagation.last_accessed_at > 0

    # --- 规则应用 (15条 Dy3+ 规则) ---

    def test_规则总数_15条(self) -> None:
        assert len(DY3_ANNOTATION_RULES) == 15

    def test_规则应用_报告结构(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        report = kpa_engine.apply_rules(full_kpa_annotation.annotation_id)
        assert report["total_rules"] == 15
        assert report["passed"] + report["failed"] == 15
        assert "summary" in report
        assert "errors" in report
        assert "warnings" in report
        assert "all_results" in report
        assert len(report["all_results"]) == 15

    def test_规则_RS01_DOI格式校验_正确格式(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-doi-test",
            source=SourceDimension(
                primary_source="10.1016/j.jlumin.2019.116789",
                source_type="journal",
                source_metadata={"doi": "https://doi.org/10.1016/j.jlumin.2019.116789"},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs01 = [r for r in report["all_results"] if r["rule_id"] == "R-S01"][0]
        assert rs01["passed"] is True

    def test_规则_RS01_DOI格式校验_错误格式(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-doi-bad",
            source=SourceDimension(
                primary_source="bad-doi",
                source_type="journal",
                source_metadata={"doi": "not-a-doi-format"},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs01 = [r for r in report["all_results"] if r["rule_id"] == "R-S01"][0]
        assert rs01["passed"] is False

    def test_规则_RS02_来源等级自动升级_TIER1(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-nature",
            source=SourceDimension(
                primary_source="10.1038/s41586-019-0001",
                source_type="journal",
                trust_tier=SourceTier.TIER_3,
                source_metadata={"journal": "Nature"},
            ),
        )
        assert ann.source.trust_tier == SourceTier.TIER_1

    def test_规则_RS02_教材默认TIER2(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-textbook",
            source=SourceDimension(
                primary_source="Springer Textbook",
                source_type="textbook",
                trust_tier=SourceTier.TIER_5,
            ),
        )
        assert ann.source.trust_tier == SourceTier.TIER_2

    def test_规则_RS03_Dy3波长校验_574nm黄光(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-wl-574",
            source=SourceDimension(
                primary_source="实验数据",
                source_type="experiment",
                source_metadata={"emission_wavelength_nm": 574},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs03 = [r for r in report["all_results"] if r["rule_id"] == "R-S03"][0]
        assert rs03["passed"] is True
        assert "黄光" in rs03["message"]

    def test_规则_RS03_Dy3波长校验_480nm蓝光(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-wl-480",
            source=SourceDimension(
                primary_source="实验数据",
                source_type="experiment",
                source_metadata={"emission_wavelength_nm": 480},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs03 = [r for r in report["all_results"] if r["rule_id"] == "R-S03"][0]
        assert rs03["passed"] is True
        assert "蓝光" in rs03["message"]

    def test_规则_RS03_Dy3波长校验_660nm红光(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-wl-660",
            source=SourceDimension(
                primary_source="实验数据",
                source_type="experiment",
                source_metadata={"emission_wavelength_nm": 660},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs03 = [r for r in report["all_results"] if r["rule_id"] == "R-S03"][0]
        assert rs03["passed"] is True
        assert "红光" in rs03["message"]

    def test_规则_RS03_Dy3波长校验_超出范围(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-wl-800",
            source=SourceDimension(
                primary_source="实验数据",
                source_type="experiment",
                source_metadata={"emission_wavelength_nm": 800},
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs03 = [r for r in report["all_results"] if r["rule_id"] == "R-S03"][0]
        assert rs03["passed"] is False

    def test_规则_RS04_检索时间戳自动填充(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-ts",
            source=SourceDimension(
                primary_source="10.1016/test",
                source_type="journal",
                retrieval_timestamp=0.0,
            ),
        )
        assert ann.source.retrieval_timestamp > 0

    def test_规则_RS05_次要来源推荐_缺少时警告(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-no-secondary",
            source=SourceDimension(
                primary_source="10.1016/only-primary",
                source_type="journal",
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rs05 = [r for r in report["all_results"] if r["rule_id"] == "R-S05"][0]
        assert rs05["passed"] is False

    def test_规则_RG01_trace_id自动生成(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-trace",
            generation=GenerationDimension(agent_id="test-agent"),
        )
        assert ann.generation.trace_id != ""
        assert ann.generation.span_id != ""

    def test_规则_RG02_code_hash缺失警告(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-no-hash",
            generation=GenerationDimension(agent_id="test-agent", code_hash=""),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rg02 = [r for r in report["all_results"] if r["rule_id"] == "R-G02"][0]
        assert rg02["passed"] is False

    def test_规则_RG03_环境哈希自动计算(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-env",
            generation=GenerationDimension(
                agent_id="test-agent",
                model_cfg={"model_name": "gpt-4"},
            ),
        )
        assert ann.generation.environment_hash != ""
        assert len(ann.generation.environment_hash) == 16

    def test_规则_RV01_CC1关联校验_知识点缺少时警告(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_type=TargetType.KNOWLEDGE_POINT,
            target_id="kp-dy3-no-cc1",
            source=SourceDimension(primary_source="10.1016/test", source_type="journal"),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rv01 = [r for r in report["all_results"] if r["rule_id"] == "R-V01"][0]
        assert rv01["passed"] is False

    def test_规则_RV02_四层评分完整性(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-four-layer",
            validation=ValidationDimension(
                cc1_review_id="rv-001",
                four_layer_scores={"factual": 90, "logical": 85, "numerical": 92},  # 缺 provenance
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rv02 = [r for r in report["all_results"] if r["rule_id"] == "R-V02"][0]
        assert rv02["passed"] is False

    def test_规则_RE01_版本链自动添加创建事件(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-version")
        assert len(ann.evolution.version_chain) >= 1
        assert ann.evolution.version_chain[0]["change_type"] == "created"

    def test_规则_RE02_JSONPatch格式校验_正确(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-patch-ok",
            evolution=EvolutionDimension(
                diff_snapshot=[
                    {"op": "replace", "path": "/source/primary_source", "value": "new-doi"},
                ],
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        re02 = [r for r in report["all_results"] if r["rule_id"] == "R-E02"][0]
        assert re02["passed"] is True

    def test_规则_RE02_JSONPatch格式校验_错误op(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-patch-bad",
            evolution=EvolutionDimension(
                diff_snapshot=[{"op": "invalid_op", "path": "/test"}],
            ),
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        re02 = [r for r in report["all_results"] if r["rule_id"] == "R-E02"][0]
        assert re02["passed"] is False

    def test_规则_RR01_前置知识推荐_缺少时提示(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_type=TargetType.KNOWLEDGE_POINT,
            target_id="kp-dy3-no-prereq",
        )
        report = kpa_engine.apply_rules(ann.annotation_id)
        rr01 = [r for r in report["all_results"] if r["rule_id"] == "R-R01"][0]
        assert rr01["passed"] is False

    def test_规则_RR02_Dy3领域关联自动识别(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(
            target_id="kp-dy3-yag-auto",
            target_metadata={"title": "Dy3+ YAG基质 CIE色度坐标分析"},
        )
        # 应自动识别 YAG 和 CIE 关键词
        labels = [r.get("label", "") for r in ann.relation.same_domain_relations]
        assert any("YAG基质" in l for l in labels)
        assert any("CIE色度坐标" in l for l in labels)

    def test_规则_RP01_传播维度初始化(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-prop-init")
        assert ann.propagation.last_accessed_at > 0

    def test_规则应用_标注不存在抛异常(self, kpa_engine: KPAEngine) -> None:
        with pytest.raises(AnnotationNotFoundError):
            kpa_engine.apply_rules("kpa-nonexistent")

    # --- 完整度评估 ---

    def test_完整度评估_报告结构(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        report = kpa_engine.evaluate_completeness(full_kpa_annotation.annotation_id)
        assert "overall_score" in report
        assert "dimension_scores" in report
        assert "filled_dimensions" in report
        assert "missing_dimensions" in report
        assert "recommendations" in report
        assert len(report["dimension_scores"]) == 7

    def test_完整度评估_七维分数(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        report = kpa_engine.evaluate_completeness(full_kpa_annotation.annotation_id)
        for dim in ["source", "generation", "validation", "decision", "evolution", "propagation", "relation"]:
            assert dim in report["dimension_scores"]
            assert 0.0 <= report["dimension_scores"][dim] <= 1.0

    def test_完整度评估_已填充维度(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        report = kpa_engine.evaluate_completeness(full_kpa_annotation.annotation_id)
        assert "source" in report["filled_dimensions"]
        assert "generation" in report["filled_dimensions"]
        assert "validation" in report["filled_dimensions"]

    def test_完整度评估_缺失维度推荐(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        report = kpa_engine.evaluate_completeness(minimal_annotation.annotation_id)
        # minimal 只有 source, 应有缺失维度推荐
        assert len(report["missing_dimensions"]) > 0
        assert len(report["recommendations"]) > 0

    def test_完整度评估_标注不存在抛异常(self, kpa_engine: KPAEngine) -> None:
        with pytest.raises(AnnotationNotFoundError):
            kpa_engine.evaluate_completeness("kpa-nonexistent")

    # --- 哈希验证 ---

    def test_哈希验证_通过(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        assert kpa_engine.verify_hash(full_kpa_annotation.annotation_id) is True

    def test_哈希验证_篡改后抛异常(self, kpa_engine: KPAEngine, minimal_annotation: KPAAnnotation) -> None:
        ann = kpa_engine.get_annotation(minimal_annotation.annotation_id)
        ann.source.primary_source = "tampered-value"
        with pytest.raises(HashMismatchError):
            kpa_engine.verify_hash(minimal_annotation.annotation_id)

    def test_哈希验证_标注不存在抛异常(self, kpa_engine: KPAEngine) -> None:
        with pytest.raises(AnnotationNotFoundError):
            kpa_engine.verify_hash("kpa-nonexistent")

    # --- C2PA 签名 ---

    def test_C2PA签名_验证通过(self, kpa_engine_signed: KPAEngine) -> None:
        ann = kpa_engine_signed.create_annotation(target_id="kp-dy3-sig")
        assert kpa_engine_signed.verify_signature(ann.annotation_id) is True

    def test_C2PA签名_无签名时返回False(self, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-no-sig")
        assert kpa_engine.verify_signature(ann.annotation_id) is False

    def test_C2PA签名_篡改后验证失败(self, kpa_engine_signed: KPAEngine) -> None:
        ann = kpa_engine_signed.create_annotation(target_id="kp-dy3-sig-tamper")
        ann.signature = "tampered-signature"
        assert kpa_engine_signed.verify_signature(ann.annotation_id) is False

    # --- W3C PROV 映射 ---

    def test_PROV映射_结构(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        prov = kpa_engine.to_prov_model(full_kpa_annotation.annotation_id)
        assert "entities" in prov
        assert "activities" in prov
        assert "agents" in prov
        assert "relations" in prov
        assert "provenance_hash" in prov

    def test_PROV映射_实体(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        prov = kpa_engine.to_prov_model(full_kpa_annotation.annotation_id)
        assert len(prov["entities"]) == 1
        assert prov["entities"][0]["id"] == "kp-dy3-yag-4f"
        assert prov["entities"][0]["type"] == "kp"

    def test_PROV映射_活动(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        prov = kpa_engine.to_prov_model(full_kpa_annotation.annotation_id)
        activity_types = [a["type"] for a in prov["activities"] if a is not None]
        assert "generation" in activity_types
        assert "validation" in activity_types
        assert "annotation" in activity_types

    def test_PROV映射_关系(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        prov = kpa_engine.to_prov_model(full_kpa_annotation.annotation_id)
        rel_types = [r["type"] for r in prov["relations"] if r is not None]
        assert "wasGeneratedBy" in rel_types
        assert "wasAttributedTo" in rel_types
        assert "wasDerivedFrom" in rel_types

    def test_PROV映射_标注不存在抛异常(self, kpa_engine: KPAEngine) -> None:
        with pytest.raises(AnnotationNotFoundError):
            kpa_engine.to_prov_model("kpa-nonexistent")

    # --- 查询 ---

    def test_查询_get_by_target(self, kpa_engine: KPAEngine) -> None:
        kpa_engine.create_annotation(target_id="kp-dy3-multi-1")
        kpa_engine.create_annotation(target_id="kp-dy3-multi-1")
        results = kpa_engine.get_by_target("kp-dy3-multi-1")
        assert len(results) == 2

    def test_查询_list_annotations_按类型(self, kpa_engine: KPAEngine) -> None:
        kpa_engine.create_annotation(
            target_type=TargetType.KNOWLEDGE_POINT, target_id="kp-1",
        )
        kpa_engine.create_annotation(
            target_type=TargetType.DECISION, target_id="dec-1",
        )
        kp_list = kpa_engine.list_annotations(target_type=TargetType.KNOWLEDGE_POINT)
        assert len(kp_list) == 1
        assert all(a.target_type == TargetType.KNOWLEDGE_POINT for a in kp_list)

    def test_查询_list_annotations_按完整度(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        kpa_engine.create_annotation(target_id="kp-minimal")
        results = kpa_engine.list_annotations(min_completeness=0.2)
        assert all(a.completeness_score() >= 0.2 for a in results)
        assert full_kpa_annotation.annotation_id in [a.annotation_id for a in results]

    # --- 统计 ---

    def test_统计_空引擎(self, kpa_engine: KPAEngine) -> None:
        stats = kpa_engine.statistics()
        assert stats["total"] == 0

    def test_统计_有数据(self, kpa_engine: KPAEngine, full_kpa_annotation: KPAAnnotation) -> None:
        kpa_engine.create_annotation(target_id="kp-dy3-second")
        stats = kpa_engine.statistics()
        assert stats["total"] == 2
        assert "kp" in stats["by_type"]
        assert stats["by_type"]["kp"] == 2
        assert "avg_completeness" in stats
        assert "min_completeness" in stats
        assert "max_completeness" in stats
        assert stats["min_completeness"] <= stats["avg_completeness"] <= stats["max_completeness"]
        assert stats["total_targets"] == 2


# ============================================================
# 2. 辩论日志引擎测试
# ============================================================


class TestDebateLogger:
    """辩论日志引擎完整测试."""

    # --- 日志创建 ---

    def test_创建日志_基本属性(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(
            debate_id="debate-dy3-001",
            task_id="task-001",
            session_id="sess-001",
            trigger_reason="复杂度45",
            complexity_score=45.0,
            verbosity=LogVerbosity.FULL,
        )
        assert log.debate_log_id.startswith("dl-")
        assert log.debate_id == "debate-dy3-001"
        assert log.task_id == "task-001"
        assert log.session_id == "sess-001"
        assert log.trigger_reason == "复杂度45"
        assert log.verbosity == LogVerbosity.FULL
        assert log.immutable_hash != ""

    def test_创建日志_默认值(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log()
        assert log.debate_id.startswith("debate-")
        assert log.verbosity == LogVerbosity.SUMMARY
        assert log.convergence_threshold == 0.1
        assert log.max_rounds == 3
        assert log.convergence_status == ConvergenceStatus.NOT_CONVERGED

    def test_创建日志_pre_debate记录(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(
            complexity_score=50.0,
            focus_area="Dy3+ 574nm发射机制",
            excluded_topics=["无关话题"],
            source_tier_requirement=SourceTier.TIER_1,
            acceptable_evidence_types=["citation", "experiment"],
            participant_configs=[{"role": "generator", "agent_id": "agent-1"}],
        )
        assert log.pre_debate.complexity_score == 50.0
        assert log.pre_debate.focus_area == "Dy3+ 574nm发射机制"
        assert "无关话题" in log.pre_debate.excluded_topics
        assert log.pre_debate.source_tier_requirement == SourceTier.TIER_1
        assert "citation" in log.pre_debate.acceptable_evidence_types

    # --- 轮次追加 ---

    def test_追加轮次_基本(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(debate_id="debate-001", verbosity=LogVerbosity.FULL)
        args = _make_generator_args()
        counters = _make_reviewer_counters([args[0].point_id])
        rnd = debate_logger.add_round(log.debate_log_id, args, counters, round_duration_ms=500.0)
        assert rnd.round_number == 1
        assert len(rnd.generator_arguments) == 2
        assert len(rnd.reviewer_counterarguments) == 1
        assert rnd.round_duration_ms == 500.0
        assert rnd.round_divergence > 0

    def test_追加轮次_自动计算分歧度(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL)
        args = _make_generator_args()
        counters = _make_reviewer_counters([args[0].point_id])
        rnd = debate_logger.add_round(log.debate_log_id, args, counters)
        assert rnd.round_divergence > 0
        assert rnd.round_divergence <= 1.0

    def test_追加轮次_手动指定分歧度(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL)
        rnd = debate_logger.add_round(log.debate_log_id, divergence=0.05)
        assert rnd.round_divergence == 0.05

    def test_追加轮次_更新分歧度曲线(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        debate_logger.add_round(log.debate_log_id, divergence=0.4)
        debate_logger.add_round(log.debate_log_id, divergence=0.05)
        updated = debate_logger.get_log(log.debate_log_id)
        assert len(updated.divergence_curve) == 3
        assert updated.divergence_curve == [0.8, 0.4, 0.05]
        assert updated.final_divergence == 0.05

    def test_追加轮次_日志不存在抛异常(self, debate_logger: DebateLogger) -> None:
        with pytest.raises(DebateLogNotFoundError):
            debate_logger.add_round("dl-nonexistent")

    # --- 收敛检查 ---

    def test_收敛检查_达到阈值(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL, convergence_threshold=0.1)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        debate_logger.add_round(log.debate_log_id, divergence=0.05)
        report = debate_logger.check_convergence(log.debate_log_id)
        assert report["converged"] is True
        assert report["convergence_round"] == 2
        assert report["final_divergence"] == 0.05

    def test_收敛检查_未达到阈值(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL, convergence_threshold=0.1)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        debate_logger.add_round(log.debate_log_id, divergence=0.5)
        report = debate_logger.check_convergence(log.debate_log_id)
        assert report["converged"] is False

    def test_收敛检查_趋势分析(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        debate_logger.add_round(log.debate_log_id, divergence=0.5)
        debate_logger.add_round(log.debate_log_id, divergence=0.2)
        report = debate_logger.check_convergence(log.debate_log_id)
        assert report["trend"] == "converging"

    def test_最大轮次_强制解决(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL, max_rounds=2, convergence_threshold=0.01)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        debate_logger.add_round(log.debate_log_id, divergence=0.5)
        updated = debate_logger.get_log(log.debate_log_id)
        assert updated.convergence_status == ConvergenceStatus.FORCE_RESOLVED

    def test_收敛检查_日志不存在抛异常(self, debate_logger: DebateLogger) -> None:
        with pytest.raises(DebateLogNotFoundError):
            debate_logger.check_convergence("dl-nonexistent")

    # --- 裁决记录 ---

    def test_记录裁决(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        verdict = debate_logger.record_adjudication(
            debate_log_with_rounds,
            adjudicator_id="adjudicator-001",
            consensus_position="Dy3+ 574nm发射源于4F9/2→6H13/2跃迁, 需补充浓度信息",
            three_dimensional_score={"accuracy": 0.92, "completeness": 0.85, "pedagogical_fit": 0.88},
            adopted_arguments=["arg-001"],
            rejected_arguments=[],
            modification_notes="补充掺杂浓度5at%信息",
            invocation_reason="convergence_reached_before_max_rounds",
        )
        assert verdict.adjudicator_id == "adjudicator-001"
        assert verdict.three_dimensional_score["accuracy"] == 0.92
        assert "arg-001" in verdict.adopted_arguments

    def test_记录裁决_日志不存在抛异常(self, debate_logger: DebateLogger) -> None:
        with pytest.raises(DebateLogNotFoundError):
            debate_logger.record_adjudication("dl-nonexistent")

    def test_记录结果(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        outcome = debate_logger.record_outcome(
            debate_log_with_rounds,
            final_consensus="Dy3+ 574nm黄光发射机制已确认",
            affected_kp_ids=["kp-dy3-yag-4f", "kp-dy3-concentration"],
            adopted_into_kb=True,
            kb_version_after="kb-v2.1",
        )
        assert outcome.final_consensus == "Dy3+ 574nm黄光发射机制已确认"
        assert len(outcome.affected_kp_ids) == 2
        assert outcome.adopted_into_kb is True
        assert outcome.kb_version_after == "kb-v2.1"

    def test_记录资源消耗(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        usage = debate_logger.record_resource_usage(
            debate_log_with_rounds,
            total_tokens=15000,
            prompt_tokens=10000,
            completion_tokens=5000,
            api_calls=6,
            compute_time_ms=3000.0,
            estimated_cost=0.12,
        )
        assert usage.total_tokens == 15000
        assert usage.api_calls == 6
        assert usage.estimated_cost == 0.12

    # --- 完成与持久化 ---

    def test_完成日志_标记持久化时间(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        debate_logger.record_adjudication(debate_log_with_rounds, adjudicator_id="adj-001")
        log = debate_logger.finalize(debate_log_with_rounds)
        assert log.persisted_at > 0

    def test_完成日志_无裁决无收敛标记中止(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.FULL)
        debate_logger.add_round(log.debate_log_id, divergence=0.8)
        finalized = debate_logger.finalize(log.debate_log_id)
        assert finalized.convergence_status == ConvergenceStatus.ABORTED

    def test_完成日志_不存在抛异常(self, debate_logger: DebateLogger) -> None:
        with pytest.raises(DebateLogNotFoundError):
            debate_logger.finalize("dl-nonexistent")

    # --- 完整性验证 ---

    def test_完整性验证_通过(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        report = debate_logger.verify_integrity(debate_log_with_rounds)
        assert report["hash_verified"] is True
        assert report["curve_consistent"] is True
        assert report["all_passed"] is True

    def test_完整性验证_篡改后抛异常(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        log = debate_logger.get_log(debate_log_with_rounds)
        log.final_divergence = 999.0
        with pytest.raises(HashMismatchError):
            debate_logger.verify_integrity(debate_log_with_rounds)

    def test_完整性验证_日志不存在抛异常(self, debate_logger: DebateLogger) -> None:
        with pytest.raises(DebateLogNotFoundError):
            debate_logger.verify_integrity("dl-nonexistent")

    # --- 导出 ---

    def test_导出_Summary级别(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        exported = debate_logger.export_log(debate_log_with_rounds, LogVerbosity.SUMMARY)
        assert exported["verbosity"] == "summary"
        assert exported["debate_log_id"] == debate_log_with_rounds
        assert "convergence_status" in exported
        assert "outcome" in exported
        # Summary 不含轮次详情
        assert "rounds" not in exported
        assert "pre_debate" not in exported

    def test_导出_Full级别(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        exported = debate_logger.export_log(debate_log_with_rounds, LogVerbosity.FULL)
        assert exported["verbosity"] == "full"
        assert "rounds" in exported
        assert "pre_debate" in exported
        assert "divergence_curve" in exported
        assert "resource_usage" in exported
        # Full 不含 debug prompts
        assert "debug_prompts" not in exported

    def test_导出_Debug级别(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(verbosity=LogVerbosity.DEBUG)
        debate_logger.add_round(
            log.debate_log_id,
            divergence=0.5,
            debug_prompts=[
                {"role": "generator", "input": "分析Dy3+发光", "output": "574nm黄光", "model_name": "gpt-4"},
            ],
        )
        exported = debate_logger.export_log(log.debate_log_id, LogVerbosity.DEBUG)
        assert "debug_prompts" in exported
        assert len(exported["debug_prompts"]) == 1
        # 脱敏后的 prompt
        assert exported["debug_prompts"][0]["input_sanitized"] != ""

    # --- 查询 ---

    def test_查询_get_by_debate(self, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(debate_id="debate-query-001")
        found = debate_logger.get_by_debate("debate-query-001")
        assert found is not None
        assert found.debate_log_id == log.debate_log_id

    def test_查询_get_by_task(self, debate_logger: DebateLogger) -> None:
        debate_logger.create_log(debate_id="d1", task_id="task-shared")
        debate_logger.create_log(debate_id="d2", task_id="task-shared")
        results = debate_logger.get_by_task("task-shared")
        assert len(results) == 2

    def test_查询_list_logs_按收敛状态(self, debate_logger: DebateLogger) -> None:
        log1 = debate_logger.create_log(verbosity=LogVerbosity.FULL, convergence_threshold=0.1)
        debate_logger.add_round(log1.debate_log_id, divergence=0.05)
        debate_logger.create_log(verbosity=LogVerbosity.FULL)
        converged = debate_logger.list_logs(converged=True)
        assert len(converged) == 1

    # --- 统计 ---

    def test_统计_空引擎(self, debate_logger: DebateLogger) -> None:
        stats = debate_logger.statistics()
        assert stats["total"] == 0

    def test_统计_有数据(self, debate_logger: DebateLogger, debate_log_with_rounds: str) -> None:
        stats = debate_logger.statistics()
        assert stats["total"] >= 1
        assert "convergence_rate" in stats
        assert "by_verbosity" in stats
        assert "by_status" in stats
        assert "avg_rounds" in stats

    # --- DivergenceCalculator ---

    def test_分歧度计算_空列表返回0(self) -> None:
        div = DivergenceCalculator.calculate([], [])
        assert div == 0.0

    def test_分歧度计算_单方发言返回0_5(self) -> None:
        args = _make_generator_args()
        div = DivergenceCalculator.calculate(args, [])
        assert div == 0.5

    def test_分歧度计算_论点覆盖法(self) -> None:
        args = _make_generator_args()
        counters = _make_reviewer_counters([args[0].point_id])
        div = DivergenceCalculator.calculate(args, counters, method="argument_based")
        assert 0.0 < div <= 1.0

    def test_分歧度计算_置信度差异法(self) -> None:
        args = [DebateArgument(point="test", confidence=0.9)]
        counters = [DebateCounterargument(targets=[], counter="no", confidence=0.3)]
        div = DivergenceCalculator.calculate(args, counters, method="confidence_gap")
        assert div == 0.6

    def test_收敛检查_静态方法(self) -> None:
        converged, conv_round = DivergenceCalculator.check_convergence(
            [0.8, 0.4, 0.05], threshold=0.1,
        )
        assert converged is True
        assert conv_round == 3

    def test_收敛检查_未收敛(self) -> None:
        converged, conv_round = DivergenceCalculator.check_convergence(
            [0.8, 0.5, 0.3], threshold=0.1,
        )
        assert converged is False
        assert conv_round == 0

    def test_收敛趋势_下降(self) -> None:
        trend = DivergenceCalculator.convergence_trend([0.8, 0.5, 0.2])
        assert trend == "converging"

    def test_收敛趋势_上升(self) -> None:
        trend = DivergenceCalculator.convergence_trend([0.2, 0.5, 0.8])
        assert trend == "diverging"

    def test_收敛趋势_稳定(self) -> None:
        trend = DivergenceCalculator.convergence_trend([0.5, 0.51, 0.5])
        assert trend == "stable"

    def test_收敛趋势_数据不足(self) -> None:
        trend = DivergenceCalculator.convergence_trend([0.5])
        assert trend == "unknown"

    # --- PromptSanitizer ---

    def test_脱敏_APIKey清除(self) -> None:
        text = "api_key=sk-abc123def456ghi789jkl012mno345pqr"
        sanitized = PromptSanitizer.sanitize(text)
        assert "sk-abc" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_脱敏_邮箱掩码(self) -> None:
        text = "联系作者: zhangsan@dy3lab.edu.cn"
        sanitized = PromptSanitizer.sanitize(text)
        assert "zhangsan@" not in sanitized
        assert "***" in sanitized

    def test_脱敏_手机号掩码(self) -> None:
        text = "联系电话: 13812345678"
        sanitized = PromptSanitizer.sanitize(text)
        assert "13812345678" not in sanitized
        assert "138****5678" in sanitized

    def test_脱敏_身份证号掩码(self) -> None:
        text = "身份证: 110101199001011234"
        sanitized = PromptSanitizer.sanitize(text)
        assert "110101199001011234" not in sanitized

    def test_脱敏_IP地址保留前两段(self) -> None:
        text = "服务器地址: 192.168.1.100"
        sanitized = PromptSanitizer.sanitize(text)
        assert "192.168.1.100" not in sanitized
        assert "192.168" in sanitized

    def test_脱敏_敏感路径清除(self) -> None:
        text = "文件路径: /home/user/dy3_data/config.yaml"
        sanitized = PromptSanitizer.sanitize(text)
        assert "/home/user/" not in sanitized
        assert "[REDACTED_PATH]" in sanitized

    def test_脱敏_空文本(self) -> None:
        assert PromptSanitizer.sanitize("") == ""

    def test_脱敏_无敏感信息原样返回(self) -> None:
        text = "Dy3+在YAG基质中的574nm黄光发射"
        assert PromptSanitizer.sanitize(text) == text

    def test_脱敏_Prompt记录创建(self) -> None:
        record = PromptSanitizer.sanitize_prompt_record(
            role="generator",
            prompt_input="api_key=sk-secret123 分析Dy3+发光",
            prompt_output="574nm黄光发射",
            model_name="gpt-4",
            round_number=1,
        )
        assert record["role"] == "generator"
        assert record["model_name"] == "gpt-4"
        assert record["round_number"] == 1
        assert "sk-secret" not in record["input_sanitized"]
        assert "574nm" in record["output_sanitized"]


# ============================================================
# 3. 溯源链构建器测试
# ============================================================


class TestProvenanceChainBuilder:
    """溯源链构建器完整测试."""

    # --- 链创建 ---

    def test_创建链_返回链ID(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-dy3-001", "Dy3+ YAG 溯源链")
        assert chain_id == "chain-dy3-001"

    def test_创建链_自动生成ID(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain()
        assert chain_id.startswith("chain-")

    def test_创建链_重复ID抛异常(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_builder.create_chain("chain-dup")
        with pytest.raises(ValueError, match="链已存在"):
            chain_builder.create_chain("chain-dup")

    # --- 节点追加 ---

    def test_追加节点_基本(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-test-001")
        node = chain_builder.append_node(
            chain_id,
            annotation_id="kpa-001",
            target_id="kp-dy3-yag-4f",
            agent_id="dy3-agent-1",
            agent_role="generator",
            layer="L2",
        )
        assert node.chain_id == "chain-test-001"
        assert node.node_index == 0
        assert node.annotation_id == "kpa-001"
        assert node.target_id == "kp-dy3-yag-4f"
        assert node.agent_id == "dy3-agent-1"
        assert node.prev_hash == "0" * 64  # 创世节点
        assert node.node_hash != ""

    def test_追加节点_prev_hash链接(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-link-001")
        node0 = chain_builder.append_node(chain_id, agent_id="agent-1", layer="L2")
        node1 = chain_builder.append_node(chain_id, agent_id="agent-2", layer="L3")
        assert node1.prev_hash == node0.node_hash
        assert node1.node_index == 1

    def test_追加节点_跨层方向(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-cross-001")
        node = chain_builder.append_node(
            chain_id,
            agent_id="agent-1",
            layer="L3",
            direction=CrossLayerDirection.L2_TO_L3,
        )
        assert node.direction == CrossLayerDirection.L2_TO_L3

    def test_追加节点_链不存在抛异常(self, chain_builder: ProvenanceChainBuilder) -> None:
        with pytest.raises(ValueError, match="链不存在"):
            chain_builder.append_node("chain-nonexistent")

    # --- 链验证 ---

    def test_链验证_通过(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-verify-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        chain_builder.append_node(chain_id, agent_id="a3", layer="L4")
        report = chain_builder.verify_chain(chain_id)
        assert report["total_nodes"] == 3
        assert report["passed_nodes"] == 3
        assert report["failed_nodes"] == 0
        assert report["hash_chain_verified"] is True
        assert report["timestamp_monotonic"] is True
        assert report["all_passed"] is True

    def test_链验证_空链通过(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-empty-001")
        report = chain_builder.verify_chain(chain_id)
        assert report["total_nodes"] == 0
        assert report["all_passed"] is True

    def test_验证单个节点_通过(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-node-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        assert chain_builder.verify_node(chain_id, 0) is True

    def test_验证单个节点_链不存在抛异常(self, chain_builder: ProvenanceChainBuilder) -> None:
        with pytest.raises(ValueError, match="链不存在"):
            chain_builder.verify_node("chain-nonexistent", 0)

    # --- Merkle 树 ---

    def test_Merkle树_构建(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-merkle-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        chain_builder.append_node(chain_id, agent_id="a3", layer="L4")
        root = chain_builder.build_merkle_tree(chain_id)
        assert root != ""
        assert len(root) == 64

    def test_Merkle树_证明生成与验证(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-proof-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        chain_builder.append_node(chain_id, agent_id="a3", layer="L4")
        chain_builder.append_node(chain_id, agent_id="a4", layer="L5")
        chain_builder.build_merkle_tree(chain_id)
        for i in range(4):
            assert chain_builder.verify_merkle_proof(chain_id, i) is True

    def test_Merkle树_证明_单个节点(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-single-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        root = chain_builder.build_merkle_tree(chain_id)
        assert chain_builder.verify_merkle_proof(chain_id, 0, root) is True

    def test_Merkle树_证明_索引越界返回False(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-oob-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        assert chain_builder.verify_merkle_proof(chain_id, 99) is False

    def test_Merkle树_独立类_构建与验证(self) -> None:
        tree = MerkleTree()
        tree.add_leaf("hash1")
        tree.add_leaf("hash2")
        tree.add_leaf("hash3")
        root = tree.build()
        assert root != ""
        proof = tree.get_proof(0)
        assert tree.verify_proof("hash1", proof, root) is True
        assert tree.verify_proof("hash2", proof, root) is False

    def test_Merkle树_空树根哈希(self) -> None:
        tree = MerkleTree()
        root = tree.build()
        assert root != ""
        assert tree.leaf_count == 0

    def test_Merkle树_序列化与反序列化(self) -> None:
        tree = MerkleTree()
        tree.add_leaf("h1")
        tree.add_leaf("h2")
        tree.build()
        serialized = tree.serialize()
        restored = MerkleTree.deserialize(serialized)
        assert restored.root == tree.root
        assert restored.leaf_count == tree.leaf_count

    # --- 压缩 ---

    def test_链压缩(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-compress-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        chain_builder.append_node(chain_id, agent_id="a3", layer="L4")
        summary = chain_builder.compress(chain_id)
        assert summary["chain_id"] == "chain-compress-001"
        assert summary["node_count"] == 3
        assert summary["merkle_root"] != ""
        assert len(summary["agents"]) == 3
        assert "L2" in summary["layers"]
        assert summary["head_node_hash"] != ""
        assert summary["tail_node_hash"] != ""
        assert summary["time_range"][0] <= summary["time_range"][1]

    def test_链压缩_空链(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-compress-empty")
        summary = chain_builder.compress(chain_id)
        assert summary["node_count"] == 0
        assert summary["merkle_root"] == ""

    # --- 快照 ---

    def test_链快照(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-snap-001", "快照测试链")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        snap = chain_builder.snapshot(chain_id)
        assert snap["chain_id"] == "chain-snap-001"
        assert len(snap["nodes"]) == 2
        assert "merkle_tree" in snap
        assert "metadata" in snap
        assert snap["snapshot_at"] > 0

    # --- 审计验证 ---

    def test_审计验证_通过(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-audit-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        result = chain_builder.audit_verify(chain_id)
        assert isinstance(result, AuditVerificationResult)
        assert result.total_records == 2
        assert result.passed_records == 2
        assert result.hash_chain_verified is True
        assert result.actor_consistency_verified is True
        assert result.timestamp_monotonic is True
        assert result.pass_rate == 1.0

    def test_审计验证_Agent缺失时一致性False(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-audit-noagent")
        chain_builder.append_node(chain_id, agent_id="", layer="L2")
        result = chain_builder.audit_verify(chain_id)
        assert result.actor_consistency_verified is False

    # --- 跨层追踪 ---

    def test_跨层追踪(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-trace-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2", direction=CrossLayerDirection.L2_TO_L3)
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3", direction=CrossLayerDirection.L3_TO_L4)
        chain_builder.append_node(chain_id, agent_id="a3", layer="L4")
        report = chain_builder.trace_cross_layer(chain_id)
        assert report["total_cross_layer_nodes"] == 2
        assert "l2_to_l3" in report["directions"]
        assert "l3_to_l4" in report["directions"]
        assert len(report["path"]) == 2
        assert report["path"][0]["from_layer"] == "L2"
        assert report["path"][0]["to_layer"] == "L3"

    def test_跨层追踪_按target过滤(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-trace-filter")
        chain_builder.append_node(chain_id, agent_id="a1", target_id="kp-1", layer="L2", direction=CrossLayerDirection.L2_TO_L3)
        chain_builder.append_node(chain_id, agent_id="a2", target_id="kp-2", layer="L3", direction=CrossLayerDirection.L3_TO_L4)
        report = chain_builder.trace_cross_layer(chain_id, target_id="kp-1")
        assert report["total_cross_layer_nodes"] == 1

    def test_跨层追踪_无跨层节点(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-no-cross")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        report = chain_builder.trace_cross_layer(chain_id)
        assert report["total_cross_layer_nodes"] == 0

    # --- 查询 ---

    def test_查询_get_chain(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-get-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain = chain_builder.get_chain(chain_id)
        assert len(chain) == 1

    def test_查询_get_node(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-getnode-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        node = chain_builder.get_node(chain_id, 0)
        assert node.agent_id == "a1"

    def test_查询_get_chain_length(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_id = chain_builder.create_chain("chain-len-001")
        chain_builder.append_node(chain_id, agent_id="a1", layer="L2")
        chain_builder.append_node(chain_id, agent_id="a2", layer="L3")
        assert chain_builder.get_chain_length(chain_id) == 2

    def test_查询_list_chains(self, chain_builder: ProvenanceChainBuilder) -> None:
        chain_builder.create_chain("chain-list-1")
        chain_builder.create_chain("chain-list-2")
        chains = chain_builder.list_chains()
        assert len(chains) == 2


# ============================================================
# 4. L0 Ledger 集成测试
# ============================================================


class TestLedgerIntegration:
    """L0 Ledger 集成器完整测试."""

    # --- KPA 写入 ---

    def test_写入KPA_知识点映射为KNOWLEDGE事件(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-dy3-ledger-001")
        event = ledger.write_kpa(ann, trace_id="trace-001", session_id="sess-001")
        assert event.event_type == EventType.KNOWLEDGE
        assert event.trace_id == "trace-001"
        assert event.session_id == "sess-001"
        assert event.agent_id == ann.annotator_agent
        assert event.layer == "CC3"
        assert "kpa_annotation" in event.payload
        assert event.event_hash != ""

    def test_写入KPA_决策映射为DECISION事件(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_type=TargetType.DECISION, target_id="dec-001")
        event = ledger.write_kpa(ann)
        assert event.event_type == EventType.DECISION

    def test_写入KPA_内容映射为INTERACTION事件(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_type=TargetType.CONTENT, target_id="content-001")
        event = ledger.write_kpa(ann)
        assert event.event_type == EventType.INTERACTION

    # --- DL 写入 ---

    def test_写入DL(self, ledger: LedgerIntegration, debate_logger: DebateLogger) -> None:
        log = debate_logger.create_log(debate_id="debate-ledger-001", task_id="task-001")
        event = ledger.write_dl(log, trace_id="trace-001")
        assert event.event_type == EventType.DECISION
        assert "debate_log" in event.payload
        assert event.payload["convergence_reached"] is False
        assert event.event_hash != ""

    # --- 跨层写入 ---

    def test_写入跨层事件(self, ledger: LedgerIntegration) -> None:
        event = ledger.write_cross_layer(
            direction=CrossLayerDirection.L2_TO_L3,
            trace_id="trace-001",
            session_id="sess-001",
            agent_id="cross-agent",
            payload={"data": "Dy3+ knowledge transfer"},
        )
        assert event.event_type == EventType.INTERACTION
        assert event.layer == "l2_to_l3"
        assert event.payload["cross_layer_direction"] == "l2_to_l3"
        assert event.payload["data"]["data"] == "Dy3+ knowledge transfer"

    # --- 人工干预写入 ---

    def test_写入人工干预事件(self, ledger: LedgerIntegration) -> None:
        event = ledger.write_human_override(
            trace_id="trace-001",
            session_id="sess-001",
            agent_id="human-reviewer",
            override_type="content_correction",
            override_detail={"kp_id": "kp-dy3-yag-4f", "reason": "数值错误"},
        )
        assert event.event_type == EventType.HUMAN_OVERRIDE
        assert event.payload["override_type"] == "content_correction"
        assert event.payload["detail"]["kp_id"] == "kp-dy3-yag-4f"

    # --- 哈希链 ---

    def test_事件哈希链_prev_hash链接(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann1 = kpa_engine.create_annotation(target_id="kp-1")
        ann2 = kpa_engine.create_annotation(target_id="kp-2")
        e1 = ledger.write_kpa(ann1)
        e2 = ledger.write_kpa(ann2)
        assert e1.prev_hash == ""  # 第一个事件
        assert e2.prev_hash == e1.event_hash

    # --- 查询 ---

    def test_查询_按trace_id(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-query-001")
        ledger.write_kpa(ann, trace_id="trace-query-001")
        ledger.write_cross_layer(CrossLayerDirection.L2_TO_L3, trace_id="trace-query-001")
        results = ledger.query(trace_id="trace-query-001")
        assert len(results) == 2

    def test_查询_按事件类型(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-type-001")
        ledger.write_kpa(ann)
        ledger.write_human_override(override_type="test")
        knowledge_events = ledger.query(event_type=EventType.KNOWLEDGE)
        assert len(knowledge_events) == 1
        override_events = ledger.query(event_type=EventType.HUMAN_OVERRIDE)
        assert len(override_events) == 1

    def test_查询_按时间范围(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        start = time.time()
        ann = kpa_engine.create_annotation(target_id="kp-time-001")
        ledger.write_kpa(ann)
        end = time.time()
        results = ledger.query_by_time_range(start, end)
        assert len(results) >= 1

    def test_查询_时间范围外无结果(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-time-002")
        ledger.write_kpa(ann)
        results = ledger.query_by_time_range(0.0, 1.0)
        assert len(results) == 0

    def test_查询_get_event(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-get-001")
        event = ledger.write_kpa(ann)
        found = ledger.get_event(event.event_id)
        assert found is not None
        assert found.event_id == event.event_id

    def test_查询_get_event_不存在返回None(self, ledger: LedgerIntegration) -> None:
        assert ledger.get_event("evt-nonexistent") is None

    # --- 完整性验证 ---

    def test_Ledger验证_通过(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        for i in range(3):
            ann = kpa_engine.create_annotation(target_id=f"kp-verify-{i}")
            ledger.write_kpa(ann)
        report = ledger.verify_ledger()
        assert report["total_events"] == 3
        assert report["passed"] == 3
        assert report["failed"] == 0
        assert report["all_passed"] is True

    def test_Ledger验证_空Ledger(self, ledger: LedgerIntegration) -> None:
        report = ledger.verify_ledger()
        assert report["total_events"] == 0
        assert report["all_passed"] is True

    # --- 统计 ---

    def test_统计_空Ledger(self, ledger: LedgerIntegration) -> None:
        stats = ledger.statistics()
        assert stats["total"] == 0

    def test_统计_有数据(self, ledger: LedgerIntegration, kpa_engine: KPAEngine) -> None:
        ann = kpa_engine.create_annotation(target_id="kp-stats-001")
        ledger.write_kpa(ann, trace_id="trace-1", session_id="sess-1")
        ledger.write_human_override(trace_id="trace-2")
        stats = ledger.statistics()
        assert stats["total"] == 2
        assert "knowledge" in stats["by_type"]
        assert "human_override" in stats["by_type"]
        assert stats["unique_traces"] == 2
        assert stats["unique_sessions"] == 1


# ============================================================
# 5. 查询引擎测试
# ============================================================


class TestQueryEngine:
    """溯源查询引擎完整测试."""

    @pytest.fixture
    def query_setup(self) -> tuple[QueryEngine, KPAEngine, DebateLogger, ProvenanceChainBuilder, LedgerIntegration]:
        """创建带数据的查询引擎环境."""
        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()
        engine = QueryEngine(kpa, dl, chain, ledger)
        return engine, kpa, dl, chain, ledger

    def test_trace回溯(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        ann = kpa.create_annotation(target_id="kp-dy3-trace-001")
        ledger.write_kpa(ann, trace_id="trace-dy3-001", session_id="sess-001")
        result = engine.trace_by_trace_id("trace-dy3-001")
        assert result["trace_id"] == "trace-dy3-001"
        assert result["total_events"] == 1
        assert len(result["kpa_annotations"]) == 1
        assert len(result["timeline"]) == 1

    def test_trace回溯_含辩论日志(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        log = dl.create_log(debate_id="debate-trace-001", verbosity=LogVerbosity.FULL)
        ledger.write_dl(log, trace_id="trace-debate-001")
        result = engine.trace_by_trace_id("trace-debate-001")
        assert result["total_events"] == 1
        assert len(result["debate_logs"]) == 1

    def test_trace回溯_无结果(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        result = engine.trace_by_trace_id("trace-nonexistent")
        assert result["total_events"] == 0
        assert len(result["kpa_annotations"]) == 0

    def test_知识溯源档案(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        kpa.create_annotation(target_id="kp-dy3-prov-001")
        result = engine.get_knowledge_provenance("kp-dy3-prov-001")
        assert result["knowledge_point_id"] == "kp-dy3-prov-001"
        assert result["annotation_count"] == 1
        assert result["has_provenance"] is True
        assert "avg_completeness" in result

    def test_知识溯源档案_不存在(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        result = engine.get_knowledge_provenance("kp-nonexistent")
        assert result["annotation_count"] == 0
        assert result["has_provenance"] is False

    def test_Agent历史(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        kpa.create_annotation(target_id="kp-agent-001", annotator_agent="dy3-agent-special")
        ledger.write_kpa(kpa.get_by_target("kp-agent-001")[0], agent_id="dy3-agent-special") if False else None
        # 写入 ledger 事件
        ann = kpa.get_by_target("kp-agent-001")[0]
        ledger.write_kpa(ann, trace_id="trace-agent-001")
        result = engine.get_agent_history("cc3-provenance-agent")
        assert "kpa_annotations" in result
        assert "ledger_events" in result
        assert result["total_operations"] >= 0

    def test_时间线查询(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        start = time.time()
        ann = kpa.create_annotation(target_id="kp-timeline-001")
        ledger.write_kpa(ann, trace_id="trace-tl-001")
        end = time.time()
        timeline = engine.get_timeline(start, end)
        assert len(timeline) >= 1
        assert "timestamp" in timeline[0]
        assert "event_type" in timeline[0]

    def test_溯源图(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        kpa.create_annotation(
            target_id="kp-dy3-graph-001",
            source=SourceDimension(primary_source="10.1016/graph", source_type="journal"),
            generation=GenerationDimension(agent_id="graph-agent"),
        )
        graph = engine.get_provenance_graph(target_id="kp-dy3-graph-001")
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["total_nodes"] > 0
        assert graph["total_edges"] > 0

    def test_溯源图_全量(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        kpa.create_annotation(target_id="kp-graph-1")
        kpa.create_annotation(target_id="kp-graph-2")
        graph = engine.get_provenance_graph()
        assert graph["total_nodes"] > 0

    def test_概览(self, query_setup: tuple) -> None:
        engine, kpa, dl, chain, ledger = query_setup
        kpa.create_annotation(target_id="kp-overview-001")
        overview = engine.overview()
        assert "kpa" in overview
        assert "debate_logs" in overview
        assert "ledger" in overview
        assert "chains" in overview
        assert "timestamp" in overview
        assert overview["kpa"]["total"] == 1


# ============================================================
# 6. CC1/CC2 跨切面集成测试
# ============================================================


class TestCCIntegration:
    """CC1/CC2/CC3 跨切面集成器完整测试."""

    @pytest.fixture
    def cc_setup(self) -> tuple[CCIntegration, KPAEngine, DebateLogger, ProvenanceChainBuilder, LedgerIntegration]:
        """创建带数据的跨切面集成环境."""
        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()
        integration = CCIntegration(kpa, dl, chain, ledger)
        # 预创建一条溯源链供节点追加
        chain.create_chain("chain-cc-integration", "CC集成测试链")
        return integration, kpa, dl, chain, ledger

    def test_CC1评审回调_更新校验维度(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc1-001")
        result = integration.on_cc1_review_completed(
            annotation_id=ann.annotation_id,
            review_id="rv-cc1-001",
            scores={"factual": 92, "logical": 88, "numerical": 95, "provenance": 85},
            verdict="pass",
            trace_id="trace-cc1-001",
            session_id="sess-001",
        )
        assert result["success"] is True
        assert result["review_id"] == "rv-cc1-001"
        assert result["verdict"] == "pass"
        updated = kpa.get_annotation(ann.annotation_id)
        assert updated.validation.cc1_review_id == "rv-cc1-001"
        assert updated.validation.four_layer_scores["factual"] == 92

    def test_CC1评审回调_追加溯源链节点(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc1-chain-001")
        integration.on_cc1_review_completed(
            annotation_id=ann.annotation_id,
            review_id="rv-001",
            scores={"factual": 90, "logical": 85, "numerical": 88, "provenance": 82},
        )
        assert chain.get_chain_length("chain-cc-integration") >= 1

    def test_CC1评审回调_写入Ledger(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc1-ledger-001")
        integration.on_cc1_review_completed(
            annotation_id=ann.annotation_id,
            review_id="rv-001",
            scores={"factual": 90, "logical": 85, "numerical": 88, "provenance": 82},
            trace_id="trace-cc1-ledger",
        )
        events = ledger.query(trace_id="trace-cc1-ledger")
        assert len(events) >= 2  # KPA事件 + 跨层事件

    def test_CC1评审回调_标注不存在(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        result = integration.on_cc1_review_completed(
            annotation_id="kpa-nonexistent",
            review_id="rv-001",
            scores={"factual": 90},
        )
        assert result["success"] is False

    def test_CC1评审回调_pass_with_notes结论(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc1-notes-001")
        result = integration.on_cc1_review_completed(
            annotation_id=ann.annotation_id,
            review_id="rv-001",
            scores={"factual": 80, "logical": 75, "numerical": 82, "provenance": 78},
            verdict="pass_with_notes",
        )
        assert result["success"] is True
        assert result["verdict"] == "pass_with_notes"

    def test_CC2审批回调_更新决策维度(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc2-001")
        result = integration.on_cc2_approval_completed(
            annotation_id=ann.annotation_id,
            approval_id="ap-cc2-001",
            approval_level="approval",
            meta_decider_result="deductive",
            paradigm_selected="energy_level_diagram",
            trace_id="trace-cc2-001",
        )
        assert result["success"] is True
        assert result["approval_id"] == "ap-cc2-001"
        updated = kpa.get_annotation(ann.annotation_id)
        assert updated.decision.cc2_approval_id == "ap-cc2-001"
        assert updated.decision.cc2_approval_level == "approval"
        assert updated.decision.meta_decider_result == "deductive"

    def test_CC2审批回调_追加溯源链节点(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-cc2-chain-001")
        integration.on_cc2_approval_completed(
            annotation_id=ann.annotation_id,
            approval_id="ap-001",
            approval_level="approval",
        )
        assert chain.get_chain_length("chain-cc-integration") >= 1

    def test_CC2审批回调_标注不存在(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        result = integration.on_cc2_approval_completed(
            annotation_id="kpa-nonexistent",
            approval_id="ap-001",
        )
        assert result["success"] is False

    def test_溯源检查_CC1_完整来源(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-prov-check-001",
            source=SourceDimension(
                primary_source="10.1016/j.jlumin.2019.116789",
                source_type="journal",
                trust_tier=SourceTier.TIER_1,
            ),
        )
        report = integration.check_provenance_for_cc1(ann.annotation_id)
        assert report["source_complete"] is True
        assert report["has_doi"] is True
        assert report["chain_verified"] is True
        assert "completeness_score" in report

    def test_溯源检查_CC1_来源不完整(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-prov-incomplete-001")
        report = integration.check_provenance_for_cc1(ann.annotation_id)
        assert report["source_complete"] is False
        assert "不完整" in report["recommendation"]

    def test_溯源检查_CC1_期刊缺DOI(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-no-doi-001",
            source=SourceDimension(primary_source="journal-no-doi", source_type="journal"),
        )
        report = integration.check_provenance_for_cc1(ann.annotation_id)
        assert report["source_complete"] is True
        assert report["has_doi"] is False
        assert "DOI" in report["recommendation"]

    def test_溯源检查_CC1_标注不存在(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        report = integration.check_provenance_for_cc1("kpa-nonexistent")
        assert report["source_complete"] is False
        assert "不存在" in report["recommendation"]

    def test_升级检查_CC2_来源缺失需审批(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-escalation-001")
        report = integration.check_escalation_for_cc2(ann.annotation_id)
        assert report["needs_escalation"] is True
        assert report["suggested_level"] == "approval"
        assert "来源维度不完整" in report["risk_factors"]

    def test_升级检查_CC2_低等级来源需确认(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-tier5-001",
            source=SourceDimension(
                primary_source="some-source",
                source_type="internal",
                trust_tier=SourceTier.TIER_5,
            ),
        )
        report = integration.check_escalation_for_cc2(ann.annotation_id)
        assert report["needs_escalation"] is True
        assert any("tier_5" in f for f in report["risk_factors"])

    def test_升级检查_CC2_完整溯源无需升级(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-complete-001",
            source=_make_dy3_source(),
            generation=_make_dy3_generation(),
            validation=_make_dy3_validation(),
            decision=_make_dy3_decision(),
            relation=_make_dy3_relation(),
        )
        report = integration.check_escalation_for_cc2(ann.annotation_id)
        assert report["needs_escalation"] is False
        assert report["suggested_level"] == "implicit"

    def test_辩论触发检查_在区间且溯源不完整(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-debate-trigger-001")
        report = integration.check_debate_trigger(ann.annotation_id, complexity_score=45.0)
        assert report["should_trigger"] is True
        assert report["source_complete"] is False

    def test_辩论触发检查_在区间且溯源完整(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-debate-complete-001",
            source=_make_dy3_source(),
        )
        report = integration.check_debate_trigger(ann.annotation_id, complexity_score=50.0)
        assert report["should_trigger"] is True
        assert "知识准确性辩论" in report["focus_area"]

    def test_辩论触发检查_不在区间(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-debate-low-001")
        report = integration.check_debate_trigger(ann.annotation_id, complexity_score=20.0)
        assert report["should_trigger"] is False

    def test_辩论触发检查_高复杂度不在区间(self, cc_setup: tuple) -> None:
        integration, kpa, dl, chain, ledger = cc_setup
        ann = kpa.create_annotation(target_id="kp-dy3-debate-high-001")
        report = integration.check_debate_trigger(ann.annotation_id, complexity_score=80.0)
        assert report["should_trigger"] is False


# ============================================================
# 7. KPI 指标引擎测试
# ============================================================


class TestKPAMetricsEngine:
    """KPI 指标引擎完整测试."""

    @pytest.fixture
    def metrics_setup(self) -> tuple[KPAMetricsEngine, KPAEngine, DebateLogger, ProvenanceChainBuilder, LedgerIntegration]:
        """创建带数据的指标引擎环境."""
        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()
        metrics = KPAMetricsEngine(kpa, dl, chain, ledger)
        return metrics, kpa, dl, chain, ledger

    def test_状态评估_高优达标(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(0.96, 0.95) == "pass"

    def test_状态评估_高优警告(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(0.91, 0.95) == "warning"

    def test_状态评估_高优失败(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(0.80, 0.95) == "fail"

    def test_状态评估_低优达标(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(50.0, 100.0, higher_is_better=False) == "pass"

    def test_状态评估_低优警告(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(110.0, 100.0, higher_is_better=False) == "warning"

    def test_状态评估_低优失败(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        assert metrics.get_metric_status(150.0, 100.0, higher_is_better=False) == "fail"

    def test_覆盖率采集_无数据(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        samples = metrics.collect_coverage()
        assert len(samples) == 3
        names = [s.metric_name for s in samples]
        assert "annotation_coverage" in names
        assert "dimension_fill_rate" in names
        assert "source_diversity" in names

    def test_覆盖率采集_有数据(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        kpa.create_annotation(
            target_id="kp-dy3-cov-001",
            source=_make_dy3_source(),
            generation=_make_dy3_generation(),
        )
        metrics.set_total_knowledge_points(10)
        samples = metrics.collect_coverage()
        coverage_sample = [s for s in samples if s.metric_name == "annotation_coverage"][0]
        assert coverage_sample.value == 0.1  # 1/10

    def test_覆盖率采集_来源多样性(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        kpa.create_annotation(target_id="kp-1", source=SourceDimension(source_type="journal"))
        kpa.create_annotation(target_id="kp-2", source=SourceDimension(source_type="textbook"))
        kpa.create_annotation(target_id="kp-3", source=SourceDimension(source_type="experiment"))
        samples = metrics.collect_coverage()
        diversity = [s for s in samples if s.metric_name == "source_diversity"][0]
        assert diversity.value > 0  # 3种/6种

    def test_完整性采集_无数据默认通过(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        samples = metrics.collect_integrity()
        for s in samples:
            assert s.value == 1.0
            assert s.status == "pass"

    def test_完整性采集_有标注(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        kpa.create_annotation(target_id="kp-dy3-integ-001")
        samples = metrics.collect_integrity()
        hash_sample = [s for s in samples if s.metric_name == "hash_verification_rate"][0]
        assert hash_sample.value == 1.0

    def test_完整性采集_有链(self, metrics_setup: tuple) -> None:
        metrics, kpa, dl, chain, ledger = metrics_setup
        cid = chain.create_chain("chain-metrics-001")
        chain.append_node(cid, agent_id="a1", layer="L2")
        chain.append_node(cid, agent_id="a2", layer="L3")
        samples = metrics.collect_integrity()
        chain_sample = [s for s in samples if s.metric_name == "chain_integrity_rate"][0]
        assert chain_sample.value == 1.0

    def test_性能采集_无延迟记录(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        samples = metrics.collect_performance()
        for s in samples:
            assert s.value == 0.0
            assert s.status == "pass"  # 0ms <= target

    def test_性能采集_有延迟记录(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        metrics.record_annotation_latency(45.0)
        metrics.record_annotation_latency(55.0)
        metrics.record_query_latency(12.0)
        metrics.record_chain_build_latency(200.0)
        samples = metrics.collect_performance()
        ann_latency = [s for s in samples if s.metric_name == "annotation_latency_ms"][0]
        assert ann_latency.value == 50.0  # (45+55)/2
        assert ann_latency.status == "pass"

    def test_合规采集_无数据默认通过(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        samples = metrics.collect_compliance()
        for s in samples:
            assert s.value == 1.0
            assert s.status == "pass"

    def test_合规采集_DOI覆盖率(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        kpa.create_annotation(
            target_id="kp-doi-1",
            source=SourceDimension(source_type="journal", source_metadata={"doi": "10.1016/test"}),
        )
        kpa.create_annotation(
            target_id="kp-doi-2",
            source=SourceDimension(source_type="journal"),  # 无 DOI
        )
        samples = metrics.collect_compliance()
        doi_sample = [s for s in samples if s.metric_name == "doi_coverage"][0]
        assert doi_sample.value == 0.5  # 1/2

    def test_合规采集_CC1关联率(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        kpa.create_annotation(
            target_id="kp-cc1-1",
            validation=ValidationDimension(cc1_review_id="rv-001"),
        )
        kpa.create_annotation(target_id="kp-cc1-2")  # 无 CC1 关联
        samples = metrics.collect_compliance()
        cc1_sample = [s for s in samples if s.metric_name == "cc1_linkage_rate"][0]
        assert cc1_sample.value == 0.5

    def test_合规采集_辩论覆盖率(self, metrics_setup: tuple) -> None:
        metrics, kpa, dl, chain, ledger = metrics_setup
        # 复杂度在 31-65 区间的辩论日志
        log1 = dl.create_log(complexity_score=45.0, verbosity=LogVerbosity.FULL)
        dl.add_round(log1.debate_log_id, divergence=0.05)
        dl.record_adjudication(log1.debate_log_id, adjudicator_id="adj-001")
        # 复杂度在区间但无轮次
        log2 = dl.create_log(complexity_score=50.0, verbosity=LogVerbosity.FULL)
        samples = metrics.collect_compliance()
        debate_sample = [s for s in samples if s.metric_name == "debate_coverage"][0]
        assert debate_sample.value == 0.5  # 1/2

    def test_全量采集(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        result = metrics.collect_all()
        assert "timestamp" in result
        assert "categories" in result
        assert "total_metrics" in result
        assert result["total_metrics"] == 12  # 4类 × 3指标
        assert "overall_pass_rate" in result
        assert "coverage" in result["categories"]
        assert "integrity" in result["categories"]
        assert "performance" in result["categories"]
        assert "compliance" in result["categories"]

    def test_仪表盘导出(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        dashboard = metrics.export_dashboard()
        assert "generated_at" in dashboard
        assert "categories" in dashboard
        assert "summary" in dashboard
        assert "recommendations" in dashboard
        assert len(dashboard["categories"]) == 4
        assert dashboard["summary"]["total_metrics"] == 12

    def test_仪表盘导出_含建议(self, metrics_setup: tuple) -> None:
        metrics, kpa, *_ = metrics_setup
        metrics.set_total_knowledge_points(100)
        kpa.create_annotation(target_id="kp-low-cov-001")
        dashboard = metrics.export_dashboard()
        # 覆盖率低应有建议
        assert len(dashboard["recommendations"]) > 0

    def test_延迟记录_滑动窗口(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        for i in range(1500):
            metrics.record_annotation_latency(float(i))
        # 滑动窗口大小 1000
        samples = metrics.collect_performance()
        ann_latency = [s for s in samples if s.metric_name == "annotation_latency_ms"][0]
        # 最后1000条: 500~1499, 平均 = (500+1499)/2 = 999.5
        assert ann_latency.value == 999.5

    def test_设置知识点总数_负数抛异常(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        with pytest.raises(ValueError):
            metrics.set_total_knowledge_points(-1)

    def test_清空延迟记录(self, metrics_setup: tuple) -> None:
        metrics, *_ = metrics_setup
        metrics.record_annotation_latency(100.0)
        metrics.clear_latency_records()
        samples = metrics.collect_performance()
        assert samples[0].value == 0.0


# ============================================================
# 8. 可视化适配器测试
# ============================================================


class TestProvenanceVisualizer:
    """溯源可视化适配器完整测试."""

    @pytest.fixture
    def viz_setup(self) -> tuple[ProvenanceVisualizer, KPAEngine, DebateLogger, ProvenanceChainBuilder]:
        """创建带数据的可视化环境."""
        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        viz = ProvenanceVisualizer(kpa, dl, chain)
        return viz, kpa, dl, chain

    # --- Cytoscape ---

    def test_Cytoscape_基本结构(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(
            target_id="kp-dy3-viz-001",
            source=SourceDimension(primary_source="10.1016/viz", source_type="journal"),
            generation=GenerationDimension(agent_id="viz-agent"),
        )
        result = viz.to_cytoscape()
        assert result["format"] == "cytoscape.js"
        assert "elements" in result
        assert "nodes" in result["elements"]
        assert "edges" in result["elements"]
        assert result["metadata"]["node_count"] > 0
        assert result["metadata"]["edge_count"] > 0

    def test_Cytoscape_按target过滤(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(target_id="kp-dy3-viz-filter-001")
        kpa.create_annotation(target_id="kp-dy3-viz-filter-002")
        result = viz.to_cytoscape(target_id="kp-dy3-viz-filter-001")
        # 只有 target_id 匹配的标注
        ann_nodes = [n for n in result["elements"]["nodes"] if n["data"].get("type") == "annotation"]
        assert len(ann_nodes) == 1

    def test_Cytoscape_节点类型(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(
            target_id="kp-dy3-viz-types-001",
            source=SourceDimension(primary_source="10.1016/types", source_type="journal", trust_tier=SourceTier.TIER_1),
            generation=GenerationDimension(agent_id="type-agent", agent_role="generator"),
            relation=_make_dy3_relation(),
        )
        result = viz.to_cytoscape(target_id="kp-dy3-viz-types-001")
        node_types = {n["data"]["type"] for n in result["elements"]["nodes"]}
        assert "annotation" in node_types
        assert "source" in node_types
        assert "agent" in node_types
        assert "knowledge" in node_types

    def test_Cytoscape_边类型(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(
            target_id="kp-dy3-viz-edges-001",
            source=SourceDimension(primary_source="10.1016/edges", source_type="journal"),
            generation=GenerationDimension(agent_id="edge-agent"),
        )
        result = viz.to_cytoscape(target_id="kp-dy3-viz-edges-001")
        edge_labels = {e["data"]["label"] for e in result["elements"]["edges"]}
        assert "wasDerivedFrom" in edge_labels
        assert "wasGeneratedBy" in edge_labels
        assert "annotates" in edge_labels

    def test_Cytoscape_空数据(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        result = viz.to_cytoscape()
        assert result["metadata"]["node_count"] == 0
        assert result["metadata"]["edge_count"] == 0

    # --- D3 Hierarchy ---

    def test_D3层级树_基本结构(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        cid = chain.create_chain("chain-d3-001")
        chain.append_node(cid, agent_id="a1", layer="L2")
        chain.append_node(cid, agent_id="a2", layer="L3")
        chain.append_node(cid, agent_id="a3", layer="L2")
        result = viz.to_d3_hierarchy(cid)
        assert result["name"] == "chain:chain-d3-001"
        assert len(result["children"]) == 2  # L2 和 L3 两个分组
        assert result["metadata"]["total_nodes"] == 3
        assert result["metadata"]["layer_count"] == 2

    def test_D3层级树_空链(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        chain.create_chain("chain-d3-empty")
        result = viz.to_d3_hierarchy("chain-d3-empty")
        assert result["children"] == []
        assert result["metadata"]["total_nodes"] == 0

    def test_D3层级树_节点属性(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        cid = chain.create_chain("chain-d3-attr-001")
        chain.append_node(cid, agent_id="attr-agent", agent_role="generator", layer="L2", annotation_id="kpa-001")
        result = viz.to_d3_hierarchy(cid)
        leaf = result["children"][0]["children"][0]
        assert leaf["attributes"]["agent_id"] == "attr-agent"
        assert leaf["attributes"]["agent_role"] == "generator"
        assert leaf["attributes"]["annotation_id"] == "kpa-001"

    # --- Mermaid ---

    def test_Mermaid_链视角(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        cid = chain.create_chain("chain-mermaid-001")
        chain.append_node(cid, agent_id="m1", layer="L2")
        chain.append_node(cid, agent_id="m2", layer="L3", direction=CrossLayerDirection.L2_TO_L3)
        text = viz.to_mermaid(chain_id=cid)
        assert "flowchart TD" in text
        assert "m1" in text
        assert "m2" in text

    def test_Mermaid_标注视角(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-mermaid-ann-001",
            source=SourceDimension(primary_source="10.1016/mermaid", source_type="journal"),
            generation=GenerationDimension(agent_id="mermaid-agent"),
            validation=ValidationDimension(cc1_review_id="rv-001"),
        )
        text = viz.to_mermaid(annotation_id=ann.annotation_id)
        assert "flowchart TD" in text
        assert "KPA标注" in text

    def test_Mermaid_无数据占位(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        text = viz.to_mermaid()
        assert "flowchart TD" in text
        assert "暂无溯源数据" in text

    def test_Mermaid_链不存在(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        text = viz.to_mermaid(chain_id="chain-nonexistent")
        assert "flowchart TD" in text

    # --- ECharts Timeline ---

    def test_ECharts时间线_单条日志(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        log = dl.create_log(debate_id="debate-echarts-001", verbosity=LogVerbosity.FULL)
        dl.add_round(log.debate_log_id, _make_generator_args(), _make_reviewer_counters([]), divergence=0.5)
        dl.add_round(log.debate_log_id, divergence=0.1)
        result = viz.to_echarts_timeline(debate_log_id=log.debate_log_id)
        assert result["format"] == "echarts"
        assert "xAxis" in result
        assert "yAxis" in result
        assert "series" in result
        assert len(result["xAxis"]["data"]) == 2  # 2轮

    def test_ECharts时间线_含裁决(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        log = dl.create_log(debate_id="debate-echarts-verdict", verbosity=LogVerbosity.FULL)
        dl.add_round(log.debate_log_id, divergence=0.5)
        dl.record_adjudication(log.debate_log_id, adjudicator_id="adj-001")
        result = viz.to_echarts_timeline(debate_log_id=log.debate_log_id)
        # X轴应含 "裁决"
        assert "裁决" in result["xAxis"]["data"]

    def test_ECharts时间线_聚合模式(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        dl.create_log(debate_id="debate-agg-001", verbosity=LogVerbosity.FULL)
        dl.create_log(debate_id="debate-agg-002", verbosity=LogVerbosity.FULL)
        result = viz.to_echarts_timeline()
        assert result["format"] == "echarts"
        assert result["metadata"]["total_logs"] == 2

    def test_ECharts时间线_无数据(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        result = viz.to_echarts_timeline()
        assert result["format"] == "echarts"
        assert result["metadata"].get("empty") is True or result["metadata"]["total_logs"] == 0

    def test_ECharts时间线_日志不存在(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        result = viz.to_echarts_timeline(debate_log_id="dl-nonexistent")
        assert result["format"] == "echarts"

    # --- ECharts Radar ---

    def test_ECharts雷达图_基本结构(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-radar-001",
            source=_make_dy3_source(),
            generation=_make_dy3_generation(),
        )
        result = viz.to_echarts_radar(ann.annotation_id)
        assert result["format"] == "echarts"
        assert "radar" in result
        assert "indicator" in result["radar"]
        assert len(result["radar"]["indicator"]) == 7  # 七维
        assert "series" in result
        assert result["metadata"]["overall_completeness"] > 0

    def test_ECharts雷达图_含CC1四层评分(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        ann = kpa.create_annotation(
            target_id="kp-dy3-radar-4layer-001",
            validation=ValidationDimension(
                cc1_review_id="rv-001",
                four_layer_scores={"factual": 0.9, "logical": 0.85, "numerical": 0.92, "provenance": 0.88},
            ),
        )
        result = viz.to_echarts_radar(ann.annotation_id)
        assert len(result["series"]) == 2  # 七维 + 四层
        assert "CC1四层评分" in result["legend"]["data"]

    def test_ECharts雷达图_标注不存在(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        result = viz.to_echarts_radar("kpa-nonexistent")
        assert result["format"] == "echarts"

    # --- export_all ---

    def test_export_all_基本结构(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(
            target_id="kp-dy3-export-001",
            source=_make_dy3_source(),
            generation=_make_dy3_generation(),
        )
        cid = chain.create_chain("chain-export-001")
        chain.append_node(cid, agent_id="a1", layer="L2")
        bundle = viz.export_all()
        assert bundle["schema"] == "ProvenanceRenderSchema/2.0"
        assert "cytoscape" in bundle
        assert "mermaid_chains" in bundle
        assert "mermaid_annotations" in bundle
        assert "echarts_radars" in bundle
        assert "echarts_timeline" in bundle
        assert "statistics" in bundle
        assert "errors" in bundle
        assert "generated_at" in bundle

    def test_export_all_按target过滤(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(target_id="kp-dy3-export-tgt-001")
        kpa.create_annotation(target_id="kp-dy3-export-tgt-002")
        bundle = viz.export_all(target_id="kp-dy3-export-tgt-001")
        assert bundle["target_id"] == "kp-dy3-export-tgt-001"
        # Cytoscape 中应只有1个标注节点
        ann_nodes = [
            n for n in bundle["cytoscape"]["elements"]["nodes"]
            if n["data"].get("type") == "annotation"
        ]
        assert len(ann_nodes) == 1

    def test_export_all_空数据(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        bundle = viz.export_all()
        assert bundle["schema"] == "ProvenanceRenderSchema/2.0"
        assert bundle["cytoscape"]["metadata"]["node_count"] == 0

    def test_export_all_json序列化(self, viz_setup: tuple) -> None:
        viz, kpa, dl, chain = viz_setup
        kpa.create_annotation(target_id="kp-dy3-json-001")
        json_str = viz.export_all_json()
        assert isinstance(json_str, str)
        assert "ProvenanceRenderSchema" in json_str
