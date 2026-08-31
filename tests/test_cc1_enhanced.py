"""CC1 四层反幻觉评审引擎 — 增强功能测试.

覆盖 ReviewPipeline 的增强特性:
1. 短路机制 (enable_short_circuit) — L1 BLOCK 时跳过后续层
2. 动态阈值 (LearnerLevel) — 不同学习者水平的阈值与层开关
3. 报告存储与检索 (store_result / get_result / list_results)
4. 评审统计 (get_statistics / pass_rate)
5. 增强自纠回路 (_generate_suggestions / _apply_corrections)
6. 新增 REST API 端点 (statistics / reports)

所有测试自包含, 可直接以 pytest 运行.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l0.cc1.review_pipeline import (
    LEARNER_LEVEL_THRESHOLDS,
    LearnerLevel,
    ReviewPipeline,
    ReviewPipelineConfig,
    ReviewResult,
)
from dy3_polaris.l0.cc1.state_machine import ReviewVerdict
from dy3_polaris.l0.cc1.layers import (
    FactLayer,
    LayerResult,
    LayerRuleResult,
    LogicLayer,
    NumericalLayer,
    ProvenanceLayer,
    ReviewLayerType,
    RuleSeverity,
)
from dy3_polaris.l0.cc1.models import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceType,
    VerificationRequest,
)
from dy3_polaris.l0.governance_router import create_governance_app


# ============================================================
# 辅助函数与 Fixtures
# ============================================================


def _make_request(
    output_text: str = "Dy3+ 的发射主峰在 575nm。",
    context_chunks: list[str] | None = None,
    agent_id: str = "agent-knowledge",
) -> VerificationRequest:
    """构造验证请求."""
    return VerificationRequest(
        agent_id=agent_id,
        output_text=output_text,
        context_chunks=context_chunks
        or ["Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"],
        citations=[],
    )


def _block_layer_result(
    layer_type: ReviewLayerType = ReviewLayerType.L1_FACT,
    detail: str = "严重事实错误, 触发阻断",
) -> LayerResult:
    """构造一个 verdict=BLOCK 的层结果 (含 CRITICAL 失败规则)."""
    return LayerResult(
        layer_type=layer_type,
        score=0.0,
        rule_results=[
            LayerRuleResult(
                rule_id="TEST-BLOCK",
                rule_name="测试阻断规则",
                passed=False,
                severity=RuleSeverity.CRITICAL,
                detail=detail,
            )
        ],
        verdict="BLOCK",
        summary=f"{layer_type.value} 触发 BLOCK",
    )


@pytest.fixture
def client():
    """创建包含评审管道的治理应用测试客户端."""
    app = create_governance_app()
    with TestClient(app) as c:
        yield c


def _ok_data(resp) -> dict:
    """提取成功响应的 data 字段, 并断言 code==0."""
    data = resp.json()
    assert data["code"] == 0, f"Expected code=0, got {data}"
    return data["data"]


# 极差的输出文本: 发射峰越界 + 错误分类 + 浓度越界
_BAD_OUTPUT = (
    "Dy3+ 的发射主峰在 650nm, 属于 d 区过渡金属, 掺杂浓度 50mol%"
)

# 高质量输出文本
_GOOD_OUTPUT = "Dy3+ 的发射主峰在 575nm, 对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁。"


# ============================================================
# 1. 短路机制 (enable_short_circuit)
# ============================================================


class TestShortCircuit:
    """短路机制: L1 返回 BLOCK 时跳过 L2/L3/L4."""

    def test_short_circuit_skips_subsequent_layers(self, monkeypatch):
        """enable_short_circuit=True 且 L1 BLOCK 时, L2/L3/L4 被跳过."""
        config = ReviewPipelineConfig(
            enable_short_circuit=True,
            enable_self_correction=False,
        )
        pipeline = ReviewPipeline(config)
        # 强制 L1 返回 BLOCK (FactLayer 规则无 CRITICAL 级别, 故用桩替换)
        monkeypatch.setattr(
            pipeline._fact_layer,
            "verify_claims",
            lambda *args, **kwargs: _block_layer_result(),
        )

        result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        assert ReviewLayerType.L1_FACT in result.layer_results
        assert ReviewLayerType.L2_LOGIC not in result.layer_results
        assert ReviewLayerType.L3_NUMERICAL not in result.layer_results
        assert ReviewLayerType.L4_PROVENANCE not in result.layer_results

    def test_no_short_circuit_evaluates_all_layers(self, monkeypatch):
        """enable_short_circuit=False 时, 即使 L1 BLOCK 也评估全部四层."""
        config = ReviewPipelineConfig(
            enable_short_circuit=False,
            enable_self_correction=False,
        )
        pipeline = ReviewPipeline(config)
        monkeypatch.setattr(
            pipeline._fact_layer,
            "verify_claims",
            lambda *args, **kwargs: _block_layer_result(),
        )

        result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        assert ReviewLayerType.L1_FACT in result.layer_results
        assert ReviewLayerType.L2_LOGIC in result.layer_results
        assert ReviewLayerType.L3_NUMERICAL in result.layer_results
        assert ReviewLayerType.L4_PROVENANCE in result.layer_results

    def test_short_circuit_layer_results_excludes_skipped(self, monkeypatch):
        """短路时 layer_results 仅包含已评估的层 (只有 L1)."""
        config = ReviewPipelineConfig(
            enable_short_circuit=True,
            enable_self_correction=False,
        )
        pipeline = ReviewPipeline(config)
        monkeypatch.setattr(
            pipeline._fact_layer,
            "verify_claims",
            lambda *args, **kwargs: _block_layer_result(),
        )

        result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        assert set(result.layer_results.keys()) == {ReviewLayerType.L1_FACT}

    def test_short_circuit_default_is_enabled(self):
        """默认配置开启短路."""
        config = ReviewPipelineConfig()
        assert config.enable_short_circuit is True


# ============================================================
# 2. 动态阈值 (LearnerLevel)
# ============================================================


class TestDynamicThresholds:
    """动态阈值: 不同学习者水平覆盖 pass/flag 阈值."""

    def test_beginner_thresholds(self):
        """BEGINNER → pass_threshold=70.0, flag_threshold=45.0."""
        config = ReviewPipelineConfig(learner_level=LearnerLevel.BEGINNER)
        assert config.pass_threshold == 70.0
        assert config.flag_threshold == 45.0

    def test_intermediate_thresholds(self):
        """INTERMEDIATE → pass_threshold=80.0, flag_threshold=55.0."""
        config = ReviewPipelineConfig(learner_level=LearnerLevel.INTERMEDIATE)
        assert config.pass_threshold == 80.0
        assert config.flag_threshold == 55.0

    def test_expert_keeps_default_thresholds(self):
        """EXPERT → 保持默认阈值 85.0 / 60.0."""
        config = ReviewPipelineConfig(learner_level=LearnerLevel.EXPERT)
        assert config.pass_threshold == 85.0
        assert config.flag_threshold == 60.0

    def test_teacher_thresholds(self):
        """TEACHER → pass_threshold=90.0, flag_threshold=70.0."""
        config = ReviewPipelineConfig(learner_level=LearnerLevel.TEACHER)
        assert config.pass_threshold == 90.0
        assert config.flag_threshold == 70.0

    def test_expert_keeps_custom_thresholds(self):
        """EXPERT 水平不覆盖用户自定义阈值."""
        config = ReviewPipelineConfig(
            learner_level=LearnerLevel.EXPERT,
            pass_threshold=92.0,
            flag_threshold=65.0,
        )
        assert config.pass_threshold == 92.0
        assert config.flag_threshold == 65.0

    def test_learner_level_thresholds_constants(self):
        """LEARNER_LEVEL_THRESHOLDS 常量值正确."""
        assert LEARNER_LEVEL_THRESHOLDS[LearnerLevel.BEGINNER][
            "pass_threshold"
        ] == 70.0
        assert LEARNER_LEVEL_THRESHOLDS[LearnerLevel.INTERMEDIATE][
            "pass_threshold"
        ] == 80.0
        assert LEARNER_LEVEL_THRESHOLDS[LearnerLevel.EXPERT][
            "pass_threshold"
        ] == 85.0
        assert LEARNER_LEVEL_THRESHOLDS[LearnerLevel.TEACHER][
            "pass_threshold"
        ] == 90.0

    def test_beginner_disables_l2_l3_l4(self):
        """BEGINNER 水平关闭 L2/L3/L4, layer_results 仅含 L1."""
        config = ReviewPipelineConfig(
            learner_level=LearnerLevel.BEGINNER,
            enable_self_correction=False,
        )
        pipeline = ReviewPipeline(config)
        result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        assert ReviewLayerType.L1_FACT in result.layer_results
        assert ReviewLayerType.L2_LOGIC not in result.layer_results
        assert ReviewLayerType.L3_NUMERICAL not in result.layer_results
        assert ReviewLayerType.L4_PROVENANCE not in result.layer_results

    def test_intermediate_disables_l4_only(self):
        """INTERMEDIATE 水平仅关闭 L4, L1/L2/L3 仍评估."""
        config = ReviewPipelineConfig(
            learner_level=LearnerLevel.INTERMEDIATE,
            enable_self_correction=False,
        )
        pipeline = ReviewPipeline(config)
        result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        assert ReviewLayerType.L1_FACT in result.layer_results
        assert ReviewLayerType.L2_LOGIC in result.layer_results
        assert ReviewLayerType.L3_NUMERICAL in result.layer_results
        assert ReviewLayerType.L4_PROVENANCE not in result.layer_results

    def test_pipeline_uses_dynamic_thresholds_in_scoring(self):
        """管道的评分引擎使用动态阈值."""
        config = ReviewPipelineConfig(learner_level=LearnerLevel.TEACHER)
        pipeline = ReviewPipeline(config)
        assert pipeline.scoring_engine.pass_threshold == 90.0
        assert pipeline.scoring_engine.flag_threshold == 70.0


# ============================================================
# 3. 报告存储与检索
# ============================================================


class TestReportStore:
    """报告存储: store_result / get_result / list_results."""

    def test_store_result_stores_review_result(self):
        """store_result 存储 ReviewResult 并返回 report_id."""
        pipeline = ReviewPipeline()
        result = ReviewResult(
            report_id="rr-store-1",
            agent_id="agent-1",
            verdict=ReviewVerdict.PASS,
        )
        rid = pipeline.store_result(result)
        assert rid == "rr-store-1"
        assert pipeline.report_store["rr-store-1"] is result

    def test_get_result_retrieves_by_report_id(self):
        """get_result 按 report_id 检索结果."""
        pipeline = ReviewPipeline()
        result = ReviewResult(
            report_id="rr-get-1",
            agent_id="agent-1",
        )
        pipeline.store_result(result)
        retrieved = pipeline.get_result("rr-get-1")
        assert retrieved is result

    def test_get_result_returns_none_for_nonexistent(self):
        """get_result 对不存在的 report_id 返回 None."""
        pipeline = ReviewPipeline()
        assert pipeline.get_result("rr-nonexistent") is None

    def test_list_results_filters_by_agent_id(self):
        """list_results 按 agent_id 过滤."""
        pipeline = ReviewPipeline()
        r1 = ReviewResult(
            report_id="rr-a1", agent_id="agent-a", verdict=ReviewVerdict.PASS
        )
        r2 = ReviewResult(
            report_id="rr-b1", agent_id="agent-b", verdict=ReviewVerdict.FLAG
        )
        pipeline.store_result(r1)
        pipeline.store_result(r2)

        results = pipeline.list_results(agent_id="agent-a")
        assert len(results) == 1
        assert results[0].agent_id == "agent-a"

    def test_list_results_filters_by_verdict(self):
        """list_results 按 verdict 过滤."""
        pipeline = ReviewPipeline()
        r1 = ReviewResult(
            report_id="rr-a1", agent_id="agent-a", verdict=ReviewVerdict.PASS
        )
        r2 = ReviewResult(
            report_id="rr-b1", agent_id="agent-b", verdict=ReviewVerdict.FLAG
        )
        r3 = ReviewResult(
            report_id="rr-c1", agent_id="agent-c", verdict=ReviewVerdict.BLOCK
        )
        pipeline.store_result(r1)
        pipeline.store_result(r2)
        pipeline.store_result(r3)

        flagged = pipeline.list_results(verdict=ReviewVerdict.FLAG)
        assert len(flagged) == 1
        assert flagged[0].verdict == ReviewVerdict.FLAG

        blocked = pipeline.list_results(verdict=ReviewVerdict.BLOCK)
        assert len(blocked) == 1
        assert blocked[0].verdict == ReviewVerdict.BLOCK

    def test_list_results_returns_all_when_no_filter(self):
        """list_results 无过滤时返回全部."""
        pipeline = ReviewPipeline()
        for i in range(3):
            pipeline.store_result(
                ReviewResult(report_id=f"rr-{i}", agent_id="agent-x")
            )
        results = pipeline.list_results()
        assert len(results) == 3

    def test_review_auto_stores_result(self):
        """review() 执行后自动存储评审结果."""
        pipeline = ReviewPipeline()
        result = pipeline.review(_make_request(output_text=_GOOD_OUTPUT))

        assert result.report_id in pipeline.report_store
        assert pipeline.get_result(result.report_id) is not None
        assert pipeline.get_result(result.report_id) is result


# ============================================================
# 4. 评审统计
# ============================================================


class TestStatistics:
    """评审统计: get_statistics / pass_rate."""

    def test_statistics_empty_when_no_reviews(self):
        """无评审时返回空统计."""
        pipeline = ReviewPipeline()
        stats = pipeline.get_statistics()
        assert stats["total"] == 0
        assert stats["pass"] == 0
        assert stats["flag"] == 0
        assert stats["block"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["avg_score"] == 0.0

    def test_statistics_counts_after_multiple_reviews(self):
        """多次评审后统计计数正确 (2 PASS + 1 BLOCK)."""
        config = ReviewPipelineConfig(enable_self_correction=False)
        pipeline = ReviewPipeline(config)

        good = _make_request(output_text=_GOOD_OUTPUT)
        bad = _make_request(output_text=_BAD_OUTPUT)

        pipeline.review(good)
        pipeline.review(good)
        pipeline.review(bad)

        stats = pipeline.get_statistics()
        assert stats["total"] == 3
        assert stats["pass"] == 2
        assert stats["block"] == 1
        assert stats["flag"] == 0

    def test_pass_rate_calculation(self):
        """pass_rate = pass / total * 100 (1 PASS + 1 BLOCK → 50.0)."""
        config = ReviewPipelineConfig(enable_self_correction=False)
        pipeline = ReviewPipeline(config)

        pipeline.review(_make_request(output_text=_GOOD_OUTPUT))
        pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        stats = pipeline.get_statistics()
        assert stats["total"] == 2
        assert stats["pass"] == 1
        assert stats["block"] == 1
        assert stats["pass_rate"] == 50.0

    def test_avg_score_calculation(self):
        """avg_score 为所有评审综合分的平均值."""
        config = ReviewPipelineConfig(enable_self_correction=False)
        pipeline = ReviewPipeline(config)

        good_result = pipeline.review(
            _make_request(output_text=_GOOD_OUTPUT)
        )
        bad_result = pipeline.review(_make_request(output_text=_BAD_OUTPUT))

        stats = pipeline.get_statistics()
        expected_avg = round(
            (good_result.composite_score + bad_result.composite_score) / 2,
            2,
        )
        assert stats["avg_score"] == expected_avg

    def test_statistics_all_pass_rate_100(self):
        """全部通过时 pass_rate=100.0."""
        config = ReviewPipelineConfig(enable_self_correction=False)
        pipeline = ReviewPipeline(config)
        pipeline.review(_make_request(output_text=_GOOD_OUTPUT))
        pipeline.review(_make_request(output_text=_GOOD_OUTPUT))

        stats = pipeline.get_statistics()
        assert stats["total"] == 2
        assert stats["pass"] == 2
        assert stats["pass_rate"] == 100.0


# ============================================================
# 增强规则检查器测试 — 消除桩实现后的验证
# ============================================================


def _make_claim_for_rule(
    text: str = "Dy3+ 的发射主峰在 575nm。",
    evidence_ids: list[str] | None = None,
) -> Claim:
    """构造声明用于规则测试."""
    return Claim(
        text=text,
        claim_type=ClaimType.FACTUAL,
        evidence_ids=evidence_ids or [],
    )


def _make_evidence_for_rule(
    content: str = "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射",
    source_uri: str = "kb://dy3/emission",
    confidence: float = 0.9,
) -> Evidence:
    """构造证据用于规则测试."""
    return Evidence(
        content=content,
        evidence_type=EvidenceType.RETRIEVED_CONTEXT,
        confidence=confidence,
        source_uri=source_uri,
    )


class TestEnhancedFactLayerRules:
    """L1 事实层增强规则测试."""

    def test_fr06_cie_coordinate_in_range(self):
        """F-R06: CIE 色坐标在正常范围内 → 通过."""
        layer = FactLayer()
        claim = _make_claim_for_rule("CIE 色度坐标 x=0.42, y=0.45。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr06 = next(r for r in result.rule_results if r.rule_id == "F-R06")
        assert fr06.passed is True

    def test_fr06_cie_coordinate_x_out_of_range(self):
        """F-R06: CIE x 坐标超出范围 → 失败."""
        layer = FactLayer()
        claim = _make_claim_for_rule("CIE 色度坐标 x=0.20, y=0.45。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr06 = next(r for r in result.rule_results if r.rule_id == "F-R06")
        assert fr06.passed is False

    def test_fr06_cie_coordinate_paren_format(self):
        """F-R06: 括号格式色坐标 (0.42, 0.45) → 通过."""
        layer = FactLayer()
        claim = _make_claim_for_rule("色度坐标 (0.42, 0.45)。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr06 = next(r for r in result.rule_results if r.rule_id == "F-R06")
        assert fr06.passed is True

    def test_fr06_cie_no_coordinate_value(self):
        """F-R06: 提到色度但无数值 → 跳过 (通过)."""
        layer = FactLayer()
        claim = _make_claim_for_rule("CIE 色度坐标是重要的参数。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr06 = next(r for r in result.rule_results if r.rule_id == "F-R06")
        assert fr06.passed is True

    def test_fr07_judd_ofelt_in_range(self):
        """F-R07: Judd-Ofelt 参数在正常范围内 → 通过."""
        layer = FactLayer()
        claim = _make_claim_for_rule("Judd-Ofelt 参数 Ω₂=5.3, Ω₄=2.1, Ω₆=1.8。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr07 = next(r for r in result.rule_results if r.rule_id == "F-R07")
        assert fr07.passed is True

    def test_fr07_judd_ofelt_out_of_range(self):
        """F-R07: Ω₂ 超出范围 → 失败."""
        layer = FactLayer()
        claim = _make_claim_for_rule("Judd-Ofelt 参数 Ω₂=15.0。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr07 = next(r for r in result.rule_results if r.rule_id == "F-R07")
        assert fr07.passed is False

    def test_fr07_judd_ofelt_numeric_form(self):
        """F-R07: 数字形式 Ω2=5.0 → 通过."""
        layer = FactLayer()
        claim = _make_claim_for_rule("Judd-Ofelt 参数 Ω2=5.0, Ω4=1.5, Ω6=1.2。")
        result = layer.verify_claim(claim, context_chunks=[])
        fr07 = next(r for r in result.rule_results if r.rule_id == "F-R07")
        assert fr07.passed is True


class TestEnhancedLogicLayerRules:
    """L2 逻辑层增强规则测试."""

    def test_lr03_4f4f_correctly_forbidden(self):
        """L-R03: 4f-4f 跃迁正确描述为禁戒 → 通过."""
        layer = LogicLayer()
        claim = _make_claim_for_rule("Dy3+ 的 4f-4f 跃迁是禁戒的, 属于弱吸收。")
        result = layer.verify_claim(claim, context_chunks=[])
        lr03 = next(r for r in result.rule_results if r.rule_id == "L-R03")
        assert lr03.passed is True

    def test_lr03_4f4f_wrongly_allowed(self):
        """L-R03: 4f-4f 跃迁被错误描述为允许 → 失败."""
        layer = LogicLayer()
        claim = _make_claim_for_rule("Dy3+ 的 4f-4f 跃迁是允许的, 属于强吸收。")
        result = layer.verify_claim(claim, context_chunks=[])
        lr03 = next(r for r in result.rule_results if r.rule_id == "L-R03")
        assert lr03.passed is False

    def test_lr03_4f5d_correctly_allowed(self):
        """L-R03: 4f-5d 跃迁正确描述为允许 → 通过."""
        layer = LogicLayer()
        claim = _make_claim_for_rule("Dy3+ 的 4f-5d 跃迁是允许的, 属于强吸收。")
        result = layer.verify_claim(claim, context_chunks=[])
        lr03 = next(r for r in result.rule_results if r.rule_id == "L-R03")
        assert lr03.passed is True

    def test_lr07_experiment_correct_order(self):
        """L-R07: 实验步骤顺序正确 → 通过."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "合成步骤: 前驱体称量 → 混合研磨 → 预烧 → 二次研磨 → 终烧 → 表征测试。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr07 = next(r for r in result.rule_results if r.rule_id == "L-R07")
        assert lr07.passed is True

    def test_lr07_experiment_wrong_order(self):
        """L-R07: 实验步骤顺序错误 (表征在烧结前) → 失败."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "合成步骤: 表征测试 → 混合研磨 → 烧结。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr07 = next(r for r in result.rule_results if r.rule_id == "L-R07")
        assert lr07.passed is False

    def test_lr09_lifetime_decreases_with_concentration(self):
        """L-R09: 浓度增加→寿命缩短 → 通过."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "随着掺杂浓度增加, 荧光寿命缩短。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr09 = next(r for r in result.rule_results if r.rule_id == "L-R09")
        assert lr09.passed is True

    def test_lr09_lifetime_increases_with_concentration(self):
        """L-R09: 浓度增加但寿命也增加 → 失败."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "随着浓度增加, 荧光寿命也增加。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr09 = next(r for r in result.rule_results if r.rule_id == "L-R09")
        assert lr09.passed is False

    def test_lr10_color_temperature_in_range(self):
        """L-R10: 色温在 3000-8000K 范围内 → 通过."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "通过调节黄蓝比实现白光发射, 色温 5000K。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr10 = next(r for r in result.rule_results if r.rule_id == "L-R10")
        assert lr10.passed is True

    def test_lr10_color_temperature_out_of_range(self):
        """L-R10: 色温超出 3000-8000K → 失败."""
        layer = LogicLayer()
        claim = _make_claim_for_rule(
            "通过调节黄蓝比实现白光发射, 色温 20000K。"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        lr10 = next(r for r in result.rule_results if r.rule_id == "L-R10")
        assert lr10.passed is False


class TestEnhancedProvenanceLayerRules:
    """L4 溯源层增强规则测试."""

    def test_pr03_high_similarity_passes(self):
        """P-R03: 引用内容与原文相似度高 → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
        )
        evidence = [
            _make_evidence_for_rule(
                content="Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
            )
        ]
        result = layer.verify_claim(claim, evidence=evidence, context_chunks=[])
        pr03 = next(r for r in result.rule_results if r.rule_id == "P-R03")
        assert pr03.passed is True

    def test_pr03_low_similarity_fails(self):
        """P-R03: 引用内容与原文相似度低 → 失败."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Dy3+ 的发射峰在 575nm, 属于黄色发射"
        )
        evidence = [
            _make_evidence_for_rule(
                content="CaTiO3 is a perovskite structure material with bandgap 3.5eV"
            )
        ]
        result = layer.verify_claim(claim, evidence=evidence, context_chunks=[])
        pr03 = next(r for r in result.rule_results if r.rule_id == "P-R03")
        assert pr03.passed is False

    def test_pr03_no_evidence_skips(self):
        """P-R03: 无证据 → 跳过 (通过)."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule("Dy3+ 发射峰 575nm")
        result = layer.verify_claim(claim, evidence=[], context_chunks=[])
        pr03 = next(r for r in result.rule_results if r.rule_id == "P-R03")
        assert pr03.passed is True

    def test_pr05_complete_traceability(self):
        """P-R05: 所有证据有 source_uri 且置信度高 → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule("Dy3+ 发射峰 575nm")
        evidence = [
            _make_evidence_for_rule(
                content="Dy3+ 发射峰 575nm",
                source_uri="kb://dy3/emission",
                confidence=0.9,
            )
        ]
        result = layer.verify_claim(claim, evidence=evidence, context_chunks=[])
        pr05 = next(r for r in result.rule_results if r.rule_id == "P-R05")
        assert pr05.passed is True

    def test_pr05_missing_source_uri(self):
        """P-R05: 证据缺少 source_uri → 失败."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule("Dy3+ 发射峰 575nm")
        evidence = [
            Evidence(
                content="Dy3+ 发射峰 575nm",
                evidence_type=EvidenceType.RETRIEVED_CONTEXT,
                confidence=0.9,
                source_uri="",  # 空 URI
            )
        ]
        result = layer.verify_claim(claim, evidence=evidence, context_chunks=[])
        pr05 = next(r for r in result.rule_results if r.rule_id == "P-R05")
        assert pr05.passed is False

    def test_pr06_recent_year_passes(self):
        """P-R06: 文献年份在 10 年内 → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Dy3+ 发射峰 575nm (Smith et al., 2023)"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        pr06 = next(r for r in result.rule_results if r.rule_id == "P-R06")
        assert pr06.passed is True

    def test_pr06_old_year_fails(self):
        """P-R06: 文献年份超过 10 年 → 失败."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Dy3+ 发射峰 575nm (Smith et al., 2005)"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        pr06 = next(r for r in result.rule_results if r.rule_id == "P-R06")
        assert pr06.passed is False

    def test_pr09_doi_format_valid(self):
        """P-R09: 引用包含 DOI → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Dy3+ 发射峰 575nm, doi:10.1234/example"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        pr09 = next(r for r in result.rule_results if r.rule_id == "P-R09")
        assert pr09.passed is True

    def test_pr09_apa_format_valid(self):
        """P-R09: APA 格式引用 → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "Smith, A.B. (2020). Dy3+ luminescence. J. Mater. Chem."
        )
        result = layer.verify_claim(claim, context_chunks=[])
        pr09 = next(r for r in result.rule_results if r.rule_id == "P-R09")
        assert pr09.passed is True

    def test_pr10_version_detected(self):
        """P-R10: 检测到版本号 → 通过."""
        layer = ProvenanceLayer()
        claim = _make_claim_for_rule(
            "知识库版本 v2.1, 已更新溯源信息"
        )
        result = layer.verify_claim(claim, context_chunks=[])
        pr10 = next(r for r in result.rule_results if r.rule_id == "P-R10")
        assert pr10.passed is True


# ============================================================
# 5. 增强自纠回路
# ============================================================


class TestSelfCorrectionSuggestions:
    """_generate_suggestions: 针对各类问题生成修正建议."""

    @pytest.mark.parametrize(
        "issue, expected_substring",
        [
            (
                "F-R01 发射峰波长校验: 发射峰波长 [650.0] 超出 570-585nm 范围",
                "570-585nm",
            ),
            (
                "F-R04 浓度猝灭阈值: 掺杂浓度 50.0mol% 超出猝灭阈值 (3-8mol%)",
                "1-5mol%",
            ),
            (
                "L-R08 分类层级逻辑: Dy3+ 被错误分类为 d 区过渡金属, 应为 f 区镧系元素",
                "镧系",
            ),
            (
                "F-R12 衰减寿命数量级: 衰减寿命 10.0ms 超出 0.1-2ms 范围",
                "0.1-2.0ms",
            ),
            (
                "F-R05 量子效率范围: 量子效率 150.0% 超出 10-85% 范围",
                "0-100%",
            ),
            (
                "F-R06 色度坐标范围: CIE 色坐标超出范围",
                "x: 0-1, y: 0-1",
            ),
            (
                "L-R10 色温-发光颜色逻辑: 色温超出范围",
                "3000-8000K",
            ),
            (
                "F-R02 能级跃迁对应校验: 能级跃迁错误",
                "⁴F₉/₂→⁶H₁₃/₂",
            ),
        ],
    )
    def test_generate_suggestions_various_issues(
        self, issue: str, expected_substring: str
    ):
        """各类问题生成对应的修正建议."""
        suggestions = ReviewPipeline._generate_suggestions([issue])
        assert len(suggestions) == 1
        assert expected_substring in suggestions[0]

    def test_generate_suggestions_multiple_issues(self):
        """多个问题生成多条建议, 数量一致."""
        issues = [
            "F-R01 发射峰波长校验: 发射峰波长 [650.0] 超出 570-585nm 范围",
            "L-R08 分类层级逻辑: Dy3+ 被错误分类为 d 区过渡金属",
        ]
        suggestions = ReviewPipeline._generate_suggestions(issues)
        assert len(suggestions) == 2

    def test_generate_suggestions_empty_issues(self):
        """空问题列表返回空建议."""
        assert ReviewPipeline._generate_suggestions([]) == []

    def test_generate_suggestions_unknown_issue_fallback(self):
        """未知问题类型使用兜底建议."""
        suggestions = ReviewPipeline._generate_suggestions(
            ["UNKNOWN-R99 未知规则: 完全无法匹配的问题描述"]
        )
        assert len(suggestions) == 1
        assert "修正问题" in suggestions[0]


class TestSelfCorrectionApply:
    """_apply_corrections: 自动修正输出文本."""

    @pytest.mark.parametrize(
        "output_text, expected_in, not_expected_in",
        [
            ("发射峰 650nm", "575nm", "650nm"),
            ("掺杂浓度 50mol%", "2mol%", "50mol%"),
            ("衰减寿命 10ms", "1.0ms", "10ms"),
            ("量子效率 150%", "50%", "150%"),
            ("属于 d 区过渡金属", "镧系", "d 区"),
        ],
    )
    def test_apply_corrections_various(
        self, output_text: str, expected_in: str, not_expected_in: str
    ):
        """各类数值/分类错误的自动修正."""
        corrected = ReviewPipeline._apply_corrections(output_text, [])
        assert expected_in in corrected
        assert not_expected_in not in corrected

    def test_apply_corrections_preserves_valid_values(self):
        """修正不会改动已在范围内的有效值."""
        text = "发射峰 575nm, 掺杂浓度 2mol%, 衰减寿命 1.0ms"
        corrected = ReviewPipeline._apply_corrections(text, [])
        assert "575nm" in corrected
        assert "2mol%" in corrected
        assert "1.0ms" in corrected

    def test_apply_corrections_combined_output(self):
        """组合多种错误的文本被一次性修正."""
        text = "Dy3+ 发射峰 650nm, 掺杂浓度 50mol%, 属于 d 区过渡金属"
        corrected = ReviewPipeline._apply_corrections(text, [])
        assert "575nm" in corrected
        assert "2mol%" in corrected
        assert "镧系" in corrected
        assert "650nm" not in corrected
        assert "d 区" not in corrected


# ============================================================
# 6. 新增 REST API 端点
# ============================================================


class TestReviewAPIEndpoints:
    """新增 API 端点: statistics / reports."""

    def test_statistics_endpoint_after_reviews(self, client: TestClient):
        """GET /review/statistics 在评审后返回统计."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": _GOOD_OUTPUT,
                "context_chunks": [
                    "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
                ],
                "citations": [],
            },
        )
        assert resp.status_code == 200
        _ok_data(resp)

        resp = client.get("/governance/v1/review/statistics")
        assert resp.status_code == 200
        stats = _ok_data(resp)
        assert stats["total"] >= 1
        assert stats["pass"] >= 1

    def test_statistics_endpoint_empty_initially(self, client: TestClient):
        """GET /review/statistics 初始无评审时返回空统计."""
        resp = client.get("/governance/v1/review/statistics")
        assert resp.status_code == 200
        stats = _ok_data(resp)
        assert stats["total"] == 0
        assert stats["pass_rate"] == 0.0

    def test_reports_endpoint_lists_reports(self, client: TestClient):
        """GET /review/reports 列出评审报告."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": _GOOD_OUTPUT,
                "context_chunks": [
                    "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
                ],
                "citations": [],
            },
        )
        result = _ok_data(resp)
        report_id = result["report_id"]

        resp = client.get("/governance/v1/review/reports")
        assert resp.status_code == 200
        reports = _ok_data(resp)
        assert isinstance(reports, list)
        assert any(r["report_id"] == report_id for r in reports)

    def test_reports_endpoint_filters_by_agent_id(self, client: TestClient):
        """GET /review/reports?agent_id= 按代理过滤."""
        client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-alpha",
                "output_text": _GOOD_OUTPUT,
                "context_chunks": [
                    "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
                ],
                "citations": [],
            },
        )
        resp = client.get(
            "/governance/v1/review/reports", params={"agent_id": "agent-alpha"}
        )
        assert resp.status_code == 200
        reports = _ok_data(resp)
        assert len(reports) >= 1
        assert all(r["agent_id"] == "agent-alpha" for r in reports)

    def test_report_endpoint_gets_specific_report(self, client: TestClient):
        """GET /review/reports/{report_id} 获取指定报告."""
        resp = client.post(
            "/governance/v1/review/execute",
            json={
                "agent_id": "agent-knowledge",
                "output_text": _GOOD_OUTPUT,
                "context_chunks": [
                    "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
                ],
                "citations": [],
            },
        )
        result = _ok_data(resp)
        report_id = result["report_id"]

        resp = client.get(f"/governance/v1/review/reports/{report_id}")
        assert resp.status_code == 200
        data = _ok_data(resp)
        assert data["report_id"] == report_id
        assert data["agent_id"] == "agent-knowledge"
        assert "layer_results" in data
        assert "layer_scores" in data

    def test_report_endpoint_nonexistent_returns_404(
        self, client: TestClient
    ):
        """GET /review/reports/{nonexistent} 返回 404."""
        resp = client.get(
            "/governance/v1/review/reports/rr-does-not-exist"
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] != 0
        assert "message" in body

    def test_statistics_reflects_multiple_reviews(self, client: TestClient):
        """多次评审后统计端点反映累计计数."""
        for _ in range(2):
            client.post(
                "/governance/v1/review/execute",
                json={
                    "agent_id": "agent-knowledge",
                    "output_text": _GOOD_OUTPUT,
                    "context_chunks": [
                        "Dy3+ 的 ⁴F₉/₂→⁶H₁₃/₂ 跃迁产生 575nm 黄色发射"
                    ],
                    "citations": [],
                },
            )
        resp = client.get("/governance/v1/review/statistics")
        stats = _ok_data(resp)
        assert stats["total"] == 2
        assert stats["pass"] == 2
        assert stats["pass_rate"] == 100.0
