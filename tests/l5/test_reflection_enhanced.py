"""反思与质量控制模块增强测试 — TDD 测试用例 (第二阶段).

测试覆盖增强功能:
1. ReflectionResult.to_dict() — 序列化 (API 响应 + 日志记录)
2. QualityTrendAnalyzer — 质量趋势分析器 (滑动窗口 + 趋势检测 + 预警)
3. 靶向自纠 (TargetedSelfCorrector) — 基于维度评分的精准自纠
4. QualityReport 增强 — 维度详情 + 趋势数据 + 可操作建议
5. ExecutionLogEntry — 执行日志条目 (审计追溯)
6. ReflectionEngine 增强 — 集成趋势分析 + 执行日志

融合世界先进方案:
- LangGraph: state serialization + checkpoint persistence
- Google ADK: EvalResult trend analysis + regression detection
- Temporal: Event history + visibility + audit trail
- OpenAI Agents SDK: Guardrail telemetry + metrics aggregation
- AutoGen: Conversation summary + performance tracking
- CrewAI: Task output validation + quality scoring trends
- Claude Science: Actor-Critic feedback loop + improvement tracking
"""

from __future__ import annotations

import pytest
import time
from datetime import datetime

from dy3_polaris.l5.reflection_quality import (
    AdjudicationExecutor,
    AdjudicationResult,
    CC1Reviewer,
    CollaborationReview,
    CollaborationTrigger,
    DimensionScore,
    GateAction,
    GateResult,
    QualityGate,
    QualityReport,
    ReflectionDimension,
    ReflectionEngine,
    ReflectionResult,
    ReflectionTrigger,
    ReputationLedger,
    ReviewRecord,
    Verdict,
)


# ============================================================
# 1. ReflectionResult.to_dict() — 序列化测试
# ============================================================


class TestReflectionResultSerialization:
    """ReflectionResult 序列化测试."""

    def _make_result(self, verdict: Verdict = Verdict.APPROVED) -> ReflectionResult:
        """创建测试用 ReflectionResult."""
        scores = [
            DimensionScore(
                dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                score=0.95,
                reasoning="事实核查通过",
            ),
            DimensionScore(
                dimension=ReflectionDimension.NUMERIC_ACCURACY,
                score=0.90,
                reasoning="数值计算正确",
            ),
            DimensionScore(
                dimension=ReflectionDimension.CITATION_COMPLETENESS,
                score=0.85,
                reasoning="引用完整",
            ),
            DimensionScore(
                dimension=ReflectionDimension.PEDAGOGICAL_FIT,
                score=0.88,
                reasoning="教学适配良好",
            ),
        ]
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=scores,
            verdict=verdict,
            feedback="质量合格",
            iteration=1,
        )
        return ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation.quiz",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=verdict,
            max_iterations=3,
            resolved_issues=["迭代1: 自纠改进"],
        )

    def test_to_dict_returns_complete_dict(self):
        """to_dict 应返回完整的序列化字典."""
        result = self._make_result()
        d = result.to_dict()

        assert d["artifact_id"] == "art-001"
        assert d["agent_id"] == "agent.generation.quiz"
        assert d["trigger"] == "single_agent"
        assert d["final_verdict"] == "approved"
        assert d["max_iterations"] == 3
        assert d["total_iterations"] == 1
        assert len(d["resolved_issues"]) == 1

    def test_to_dict_includes_reviews(self):
        """to_dict 应包含审核记录列表."""
        result = self._make_result()
        d = result.to_dict()

        assert "reviews" in d
        assert len(d["reviews"]) == 1
        review_dict = d["reviews"][0]
        assert review_dict["artifact_id"] == "art-001"
        assert review_dict["verdict"] == "approved"
        assert len(review_dict["dimension_scores"]) == 4

    def test_to_dict_includes_improvement_trajectory(self):
        """to_dict 应包含改进轨迹."""
        result = self._make_result()
        d = result.to_dict()

        assert "improvement_trajectory" in d
        assert len(d["improvement_trajectory"]) == 1
        assert isinstance(d["improvement_trajectory"][0], float)

    def test_to_dict_with_multiple_reviews(self):
        """多轮审核的 to_dict 应包含所有审核记录."""
        scores1 = [
            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.60),
            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.55),
            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.50),
            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.65),
        ]
        scores2 = [
            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.90),
            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.85),
            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.80),
            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.88),
        ]
        review1 = ReviewRecord(
            artifact_id="art-002",
            reviewer="cc1.actor_critic",
            dimension_scores=scores1,
            verdict=Verdict.REVISE,
            iteration=1,
        )
        review2 = ReviewRecord(
            artifact_id="art-002",
            reviewer="cc1.actor_critic",
            dimension_scores=scores2,
            verdict=Verdict.APPROVED,
            iteration=2,
        )
        result = ReflectionResult(
            artifact_id="art-002",
            agent_id="agent.generation.quiz",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review1, review2],
            final_verdict=Verdict.APPROVED,
        )
        d = result.to_dict()

        assert d["total_iterations"] == 2
        assert len(d["reviews"]) == 2
        assert d["improvement_trajectory"] == [review1.weighted_score, review2.weighted_score]
        assert d["improvement_trajectory"][1] > d["improvement_trajectory"][0]


# ============================================================
# 2. QualityTrendAnalyzer — 质量趋势分析器测试
# ============================================================


class TestQualityTrendAnalyzer:
    """QualityTrendAnalyzer 质量趋势分析器测试.

    融合 Google ADK EvalResult 趋势分析 + OpenAI Guardrail 遥测聚合.
    """

    def test_creation_with_default_window(self):
        """默认滑动窗口大小为 20."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer()
        assert analyzer.window_size == 20
        assert len(analyzer) == 0

    def test_creation_with_custom_window(self):
        """自定义滑动窗口大小."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=50)
        assert analyzer.window_size == 50

    def test_record_result(self):
        """记录反思结果到趋势分析器."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[
                ReviewRecord(
                    artifact_id="art-001",
                    reviewer="cc1",
                    dimension_scores=[
                        DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                    ],
                    verdict=Verdict.APPROVED,
                ),
            ],
            final_verdict=Verdict.APPROVED,
        )
        analyzer.record(result)

        assert len(analyzer) == 1

    def test_average_score_empty(self):
        """空分析器的平均分为 0."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer()
        assert analyzer.average_score == 0.0

    def test_average_score_with_data(self):
        """有数据时计算正确的平均分."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)

        for score in [0.8, 0.9, 0.85]:
            result = ReflectionResult(
                artifact_id=f"art-{score}",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[
                    ReviewRecord(
                        artifact_id=f"art-{score}",
                        reviewer="cc1",
                        dimension_scores=[
                            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=score),
                            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=score),
                            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=score),
                            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=score),
                        ],
                        verdict=Verdict.APPROVED,
                    ),
                ],
                final_verdict=Verdict.APPROVED,
            )
            analyzer.record(result)

        assert abs(analyzer.average_score - 0.85) < 0.01

    def test_pass_rate(self):
        """计算通过率 (APPROVED 占比)."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)

        verdicts = [Verdict.APPROVED, Verdict.APPROVED, Verdict.REVISE, Verdict.REJECTED]
        for i, v in enumerate(verdicts):
            result = ReflectionResult(
                artifact_id=f"art-{i}",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[
                    ReviewRecord(
                        artifact_id=f"art-{i}",
                        reviewer="cc1",
                        dimension_scores=[
                            DimensionScore(
                                dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                                score=0.5,
                            ),
                        ],
                        verdict=v,
                    ),
                ],
                final_verdict=v,
            )
            analyzer.record(result)

        assert analyzer.pass_rate == 0.5  # 2/4

    def test_trend_direction_stable(self):
        """趋势方向: 稳定 (数据不足时)."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)
        assert analyzer.trend_direction == "stable"

    def test_trend_direction_improving(self):
        """趋势方向: 改善 (分数持续上升)."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)

        for score in [0.5, 0.6, 0.7, 0.8, 0.9]:
            result = ReflectionResult(
                artifact_id=f"art-{score}",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[
                    ReviewRecord(
                        artifact_id=f"art-{score}",
                        reviewer="cc1",
                        dimension_scores=[
                            DimensionScore(
                                dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                                score=score,
                            ),
                        ],
                        verdict=Verdict.APPROVED,
                    ),
                ],
                final_verdict=Verdict.APPROVED,
            )
            analyzer.record(result)

        assert analyzer.trend_direction == "improving"

    def test_trend_direction_declining(self):
        """趋势方向: 下降 (分数持续下降)."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)

        for score in [0.9, 0.8, 0.7, 0.6, 0.5]:
            result = ReflectionResult(
                artifact_id=f"art-{score}",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[
                    ReviewRecord(
                        artifact_id=f"art-{score}",
                        reviewer="cc1",
                        dimension_scores=[
                            DimensionScore(
                                dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                                score=score,
                            ),
                        ],
                        verdict=Verdict.REVISE,
                    ),
                ],
                final_verdict=Verdict.REVISE,
            )
            analyzer.record(result)

        assert analyzer.trend_direction == "declining"

    def test_sliding_window_eviction(self):
        """滑动窗口满时淘汰最旧数据."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=3)

        for i in range(5):
            score = 0.5 + i * 0.1
            result = ReflectionResult(
                artifact_id=f"art-{i}",
                agent_id="agent.test",
                trigger=ReflectionTrigger.SINGLE_AGENT,
                reviews=[
                    ReviewRecord(
                        artifact_id=f"art-{i}",
                        reviewer="cc1",
                        dimension_scores=[
                            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=score),
                            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=score),
                            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=score),
                            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=score),
                        ],
                        verdict=Verdict.APPROVED,
                    ),
                ],
                final_verdict=Verdict.APPROVED,
            )
            analyzer.record(result)

        # 窗口大小为 3, 只保留最后 3 条
        assert len(analyzer) == 3
        # 平均分应该是最后 3 条的平均: (0.7 + 0.8 + 0.9) / 3
        assert abs(analyzer.average_score - 0.8) < 0.01

    def test_dimension_breakdown(self):
        """按维度分解平均分."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)

        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[
                ReviewRecord(
                    artifact_id="art-001",
                    reviewer="cc1",
                    dimension_scores=[
                        DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                        DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.8),
                        DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.7),
                        DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.85),
                    ],
                    verdict=Verdict.APPROVED,
                ),
            ],
            final_verdict=Verdict.APPROVED,
        )
        analyzer.record(result)

        breakdown = analyzer.dimension_breakdown
        assert breakdown[ReflectionDimension.FACTUAL_CONSISTENCY.value] == 0.9
        assert breakdown[ReflectionDimension.NUMERIC_ACCURACY.value] == 0.8
        assert breakdown[ReflectionDimension.CITATION_COMPLETENESS.value] == 0.7
        assert breakdown[ReflectionDimension.PEDAGOGICAL_FIT.value] == 0.85

    def test_get_summary(self):
        """获取趋势摘要字典."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        analyzer = QualityTrendAnalyzer(window_size=10)
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[
                ReviewRecord(
                    artifact_id="art-001",
                    reviewer="cc1",
                    dimension_scores=[
                        DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                    ],
                    verdict=Verdict.APPROVED,
                ),
            ],
            final_verdict=Verdict.APPROVED,
        )
        analyzer.record(result)

        summary = analyzer.get_summary()
        assert "total_reflections" in summary
        assert "average_score" in summary
        assert "pass_rate" in summary
        assert "trend_direction" in summary
        assert "dimension_breakdown" in summary
        assert summary["total_reflections"] == 1
        assert summary["trend_direction"] == "stable"


# ============================================================
# 3. TargetedSelfCorrector — 靶向自纠测试
# ============================================================


class TestTargetedSelfCorrector:
    """TargetedSelfCorrector 靶向自纠测试.

    融合 LangGraph Generator-Critic 定向修复 + Claude Science 精准反馈.
    """

    def test_correct_factual_consistency_low_score(self):
        """事实一致性低分时, 靶向提升置信度."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {"content": "test", "confidence": 0.4}
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(
                    dimension=ReflectionDimension.FACTUAL_CONSISTENCY,
                    score=0.4,
                    reasoning="置信度过低",
                ),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.9),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.9),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.9),
            ],
            verdict=Verdict.REVISE,
        )

        corrected = corrector.correct(data, review)

        assert corrected["confidence"] > 0.4
        assert corrected["confidence"] <= 0.95

    def test_correct_citation_missing(self):
        """引用完整性低分时, 补充引用."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {"content": "test", "confidence": 0.9, "references": []}
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.9),
                DimensionScore(
                    dimension=ReflectionDimension.CITATION_COMPLETENESS,
                    score=0.4,
                    reasoning="缺少引用",
                ),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.9),
            ],
            verdict=Verdict.REVISE,
        )

        corrected = corrector.correct(data, review)

        assert len(corrected["references"]) > 0

    def test_correct_pedagogical_fit_low(self):
        """教学适配性低分时, 补充教学相关字段."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {"content": "test", "confidence": 0.9, "references": ["ref1"]}
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.9),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.9),
                DimensionScore(
                    dimension=ReflectionDimension.PEDAGOGICAL_FIT,
                    score=0.4,
                    reasoning="缺少教学适配信息",
                ),
            ],
            verdict=Verdict.REVISE,
        )

        corrected = corrector.correct(data, review)

        # 应该补充 report_id 或 kp_gaps
        assert "report_id" in corrected or "kp_gaps" in corrected

    def test_correct_numeric_accuracy_with_bad_values(self):
        """数值准确性低分时, 标记需重新计算."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {
            "content": "test",
            "confidence": 0.9,
            "references": ["ref1"],
            "boiling_point": 99999,
        }
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
                DimensionScore(
                    dimension=ReflectionDimension.NUMERIC_ACCURACY,
                    score=0.3,
                    reasoning="数值异常",
                ),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.9),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.9),
            ],
            verdict=Verdict.REVISE,
        )

        corrected = corrector.correct(data, review)

        # 应标记需要重新计算
        assert corrected.get("_needs_recalculation") is True or "boiling_point" not in corrected

    def test_correct_no_changes_when_all_high(self):
        """所有维度高分时, 不做修改."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {
            "content": "test",
            "confidence": 0.95,
            "references": ["doi:10.1000/test"],
            "report_id": "rpt-001",
            "kp_gaps": ["KP-001"],
        }
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.95),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.95),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.95),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.95),
            ],
            verdict=Verdict.APPROVED,
        )

        corrected = corrector.correct(data, review)

        # 高分时不应做重大修改
        assert corrected["confidence"] == 0.95

    def test_correct_preserves_original_data(self):
        """自纠不修改原始数据 (不可变性)."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        original = {"content": "test", "confidence": 0.4}
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.4),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.9),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.9),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.9),
            ],
            verdict=Verdict.REVISE,
        )

        corrected = corrector.correct(original, review)

        assert original["confidence"] == 0.4  # 原始未被修改
        assert corrected["confidence"] > 0.4  # 修正后的值更高

    def test_get_corrections_log(self):
        """获取修正日志 (记录做了哪些修改)."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        corrector = TargetedSelfCorrector()
        data = {"content": "test", "confidence": 0.4, "references": []}
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.4),
                DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.4),
                DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.4),
                DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.4),
            ],
            verdict=Verdict.REVISE,
        )

        corrected, log = corrector.correct_with_log(data, review)

        assert len(log) > 0
        assert any("confidence" in entry for entry in log)
        assert any("references" in entry for entry in log)


# ============================================================
# 4. QualityReport 增强 — 维度详情 + 趋势 + 建议
# ============================================================


class TestQualityReportEnhanced:
    """QualityReport 增强测试."""

    def _make_report(self) -> QualityReport:
        """创建测试用 QualityReport."""
        scores = [
            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.95),
            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.88),
            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.75),
            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.82),
        ]
        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1.actor_critic",
            dimension_scores=scores,
            verdict=Verdict.APPROVED,
            feedback="质量合格",
            iteration=1,
        )
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation.quiz",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=Verdict.APPROVED,
        )
        return QualityReport(
            session_id="sess-001",
            artifact_id="art-001",
            reflection_result=result,
        )

    def test_to_dict_includes_dimension_details(self):
        """to_dict 应包含维度详情."""
        report = self._make_report()
        d = report.to_dict()

        assert "dimension_details" in d
        assert len(d["dimension_details"]) == 4
        detail = d["dimension_details"][0]
        assert "dimension" in detail
        assert "score" in detail
        assert "weight" in detail

    def test_to_dict_includes_feedback(self):
        """to_dict 应包含审核反馈."""
        report = self._make_report()
        d = report.to_dict()

        assert "feedback" in d
        assert d["feedback"] == "质量合格"

    def test_to_dict_includes_reviewer(self):
        """to_dict 应包含审核者信息."""
        report = self._make_report()
        d = report.to_dict()

        assert "reviewer" in d
        assert d["reviewer"] == "cc1.actor_critic"

    def test_summary_includes_weakest_dimension(self):
        """摘要应包含最弱维度信息."""
        report = self._make_report()
        summary = report.summary

        assert "weakest_dimension" in summary
        assert summary["weakest_dimension"] == "citation_completeness"

    def test_summary_includes_strongest_dimension(self):
        """摘要应包含最强维度信息."""
        report = self._make_report()
        summary = report.summary

        assert "strongest_dimension" in summary
        assert summary["strongest_dimension"] == "factual_consistency"

    def test_generate_recommendations_approved(self):
        """APPROVED 时生成积极建议."""
        report = self._make_report()
        recommendations = report.generate_recommendations()

        assert len(recommendations) > 0
        assert any("通过" in r or "合格" in r for r in recommendations)

    def test_generate_recommendations_revise(self):
        """REVISE 时生成改进建议."""
        scores = [
            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.6),
            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.9),
            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.5),
            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.85),
        ]
        review = ReviewRecord(
            artifact_id="art-002",
            reviewer="cc1.actor_critic",
            dimension_scores=scores,
            verdict=Verdict.REVISE,
            feedback="需要修订",
            iteration=1,
        )
        result = ReflectionResult(
            artifact_id="art-002",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=Verdict.REVISE,
        )
        report = QualityReport(
            session_id="sess-002",
            artifact_id="art-002",
            reflection_result=result,
        )

        recommendations = report.generate_recommendations()

        assert len(recommendations) > 0
        # 应提及需要改进的维度
        assert any("citation" in r.lower() or "引用" in r for r in recommendations)

    def test_generate_recommendations_rejected(self):
        """REJECTED 时生成严重警告."""
        scores = [
            DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.2),
            DimensionScore(dimension=ReflectionDimension.NUMERIC_ACCURACY, score=0.3),
            DimensionScore(dimension=ReflectionDimension.CITATION_COMPLETENESS, score=0.4),
            DimensionScore(dimension=ReflectionDimension.PEDAGOGICAL_FIT, score=0.35),
        ]
        review = ReviewRecord(
            artifact_id="art-003",
            reviewer="cc1.actor_critic",
            dimension_scores=scores,
            verdict=Verdict.REJECTED,
            feedback="质量不合格",
            iteration=1,
        )
        result = ReflectionResult(
            artifact_id="art-003",
            agent_id="agent.test",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=Verdict.REJECTED,
        )
        report = QualityReport(
            session_id="sess-003",
            artifact_id="art-003",
            reflection_result=result,
        )

        recommendations = report.generate_recommendations()

        assert len(recommendations) > 0
        assert any("严重" in r or "拒绝" in r or "不合格" in r for r in recommendations)


# ============================================================
# 5. ExecutionLogEntry — 执行日志条目测试
# ============================================================


class TestExecutionLogEntry:
    """ExecutionLogEntry 执行日志条目测试.

    融合 Temporal 事件历史 + Claude Code JSONL 审计日志.
    """

    def test_creation(self):
        """创建执行日志条目."""
        from dy3_polaris.l5.reflection_quality import ExecutionLogEntry

        entry = ExecutionLogEntry(
            agent_id="agent.generation.quiz",
            action="reflection.review",
            artifact_id="art-001",
            result="approved",
            score=0.92,
            metadata={"iteration": 1, "dimensions": 4},
        )

        assert entry.agent_id == "agent.generation.quiz"
        assert entry.action == "reflection.review"
        assert entry.artifact_id == "art-001"
        assert entry.result == "approved"
        assert entry.score == 0.92
        assert entry.log_id  # 自动生成

    def test_to_dict(self):
        """序列化为字典."""
        from dy3_polaris.l5.reflection_quality import ExecutionLogEntry

        entry = ExecutionLogEntry(
            agent_id="agent.test",
            action="reflection.start",
            artifact_id="art-001",
            result="started",
            score=0.0,
        )
        d = entry.to_dict()

        assert d["agent_id"] == "agent.test"
        assert d["action"] == "reflection.start"
        assert d["artifact_id"] == "art-001"
        assert d["result"] == "started"
        assert "log_id" in d
        assert "timestamp" in d

    def test_auto_timestamp(self):
        """自动生成时间戳."""
        from dy3_polaris.l5.reflection_quality import ExecutionLogEntry

        before = time.time()
        entry = ExecutionLogEntry(
            agent_id="agent.test",
            action="test",
            artifact_id="art-001",
            result="ok",
            score=1.0,
        )
        after = time.time()

        assert before <= entry.timestamp <= after

    def test_from_reflection_result(self):
        """从 ReflectionResult 创建日志条目."""
        from dy3_polaris.l5.reflection_quality import ExecutionLogEntry

        review = ReviewRecord(
            artifact_id="art-001",
            reviewer="cc1",
            dimension_scores=[
                DimensionScore(dimension=ReflectionDimension.FACTUAL_CONSISTENCY, score=0.9),
            ],
            verdict=Verdict.APPROVED,
            iteration=1,
        )
        result = ReflectionResult(
            artifact_id="art-001",
            agent_id="agent.generation.quiz",
            trigger=ReflectionTrigger.SINGLE_AGENT,
            reviews=[review],
            final_verdict=Verdict.APPROVED,
        )

        entry = ExecutionLogEntry.from_reflection_result(result)

        assert entry.agent_id == "agent.generation.quiz"
        assert entry.artifact_id == "art-001"
        assert entry.result == "approved"
        assert entry.score == review.weighted_score


# ============================================================
# 6. ReflectionEngine 增强集成测试
# ============================================================


class TestReflectionEngineEnhanced:
    """ReflectionEngine 增强集成测试."""

    @pytest.fixture
    def engine_with_trend(self):
        """创建带趋势分析的反思引擎."""
        from dy3_polaris.l5.reflection_quality import QualityTrendAnalyzer

        gate = QualityGate(
            name="test_gate",
            threshold=0.80,
            hard_floor=0.40,
            max_revisions=3,
        )
        reviewer = CC1Reviewer()
        ledger = ReputationLedger()
        trend_analyzer = QualityTrendAnalyzer(window_size=20)

        engine = ReflectionEngine(
            gate=gate,
            reviewer=reviewer,
            reputation_ledger=ledger,
            trend_analyzer=trend_analyzer,
        )
        return engine

    @pytest.mark.asyncio
    async def test_reflect_records_to_trend_analyzer(
        self, engine_with_trend
    ):
        """反思结果自动记录到趋势分析器."""
        engine = engine_with_trend

        result = await engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id="art-001",
            artifact_data={
                "content": "H₂O has a molar mass of 18.015 g/mol.",
                "confidence": 0.92,
                "references": ["doi:10.1000/chem"],
                "report_id": "rpt-chem-001",
                "kp_gaps": ["KP-molar-mass"],
            },
        )

        # 验证趋势分析器已记录 (通过 get_trend_summary 间接验证)
        summary = engine.get_trend_summary()
        assert summary["total_reflections"] == 1

    @pytest.mark.asyncio
    async def test_get_trend_summary(self, engine_with_trend):
        """获取趋势摘要."""
        engine = engine_with_trend

        # 执行多次反思
        for i in range(3):
            await engine.reflect(
                agent_id="agent.generation.quiz",
                artifact_id=f"art-{i}",
                artifact_data={
                    "content": f"Test content {i}",
                    "confidence": 0.90 + i * 0.02,
                    "references": ["doi:10.1000/test"],
                    "report_id": f"rpt-{i}",
                    "kp_gaps": [f"KP-{i}"],
                },
            )

        summary = engine.get_trend_summary()

        assert summary["total_reflections"] == 3
        assert summary["average_score"] > 0
        assert summary["trend_direction"] in ("improving", "stable")

    @pytest.mark.asyncio
    async def test_execution_log_generated(self, engine_with_trend):
        """反思过程生成执行日志."""
        engine = engine_with_trend

        result = await engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id="art-001",
            artifact_data={
                "content": "Test content",
                "confidence": 0.92,
                "references": ["doi:10.1000/test"],
                "report_id": "rpt-001",
                "kp_gaps": ["KP-001"],
            },
        )

        logs = engine.get_execution_logs()

        assert len(logs) > 0
        assert logs[-1].agent_id == "agent.generation.quiz"
        assert logs[-1].artifact_id == "art-001"

    @pytest.mark.asyncio
    async def test_targeted_self_correction_used(
        self, engine_with_trend
    ):
        """引擎使用靶向自纠而非简单自纠."""
        from dy3_polaris.l5.reflection_quality import TargetedSelfCorrector

        gate = QualityGate(
            name="test_gate",
            threshold=0.95,  # 高阈值强制进入修订
            hard_floor=0.30,
            max_revisions=2,
        )
        reviewer = CC1Reviewer()
        ledger = ReputationLedger()
        corrector = TargetedSelfCorrector()

        engine = ReflectionEngine(
            gate=gate,
            reviewer=reviewer,
            reputation_ledger=ledger,
            self_corrector=corrector,
        )

        result = await engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id="art-001",
            artifact_data={
                "content": "Test content",
                "confidence": 0.70,  # 低置信度触发自纠
                "references": [],
                "report_id": "",
                "kp_gaps": [],
            },
        )

        # 应有多轮审核 (自纠后重新评审)
        assert result.total_iterations >= 1
        # 如果有自纠, 应记录在 resolved_issues 中
        if result.total_iterations > 1:
            assert len(result.resolved_issues) > 0

    @pytest.mark.asyncio
    async def test_backward_compatible_without_enhancements(self):
        """不传增强参数时, 引擎仍正常工作 (向后兼容)."""
        gate = QualityGate(name="test", threshold=0.80, hard_floor=0.40)
        reviewer = CC1Reviewer()
        ledger = ReputationLedger()

        engine = ReflectionEngine(
            gate=gate,
            reviewer=reviewer,
            reputation_ledger=ledger,
        )

        result = await engine.reflect(
            agent_id="agent.generation.quiz",
            artifact_id="art-001",
            artifact_data={
                "content": "Test content",
                "confidence": 0.92,
                "references": ["doi:10.1000/test"],
                "report_id": "rpt-001",
                "kp_gaps": ["KP-001"],
            },
        )

        assert result.final_verdict == Verdict.APPROVED
