"""CC1 四层反幻觉评审引擎 — 增强评审管道.

编排四层评审 (L1-L4) + 状态机 + 自纠回路 + 综合评分,
形成完整的反幻觉评审流水线.

管道流程::

    VerificationRequest
        ↓
    ┌─ 声明提取 (ClaimExtractor)
    ├─ 证据收集 (EvidenceCollector)
    ├─ L1 事实层评审 → LayerResult
    ├─ L2 逻辑层评审 → LayerResult
    ├─ L3 数值层评审 → LayerResult
    ├─ L4 溯源层评审 → LayerResult
    ├─ 综合评分 (CompositeScoringEngine)
    ├─ 判决 (PASS / FLAG / BLOCK)
    └─ 自纠回路 (if FLAG/BLOCK)
        ↓
    ReviewResult

融合世界先进方案:
- CoVe (Chain-of-Verification): 自纠回路验证-修正-再验证
- RAGAS: 声明级粒度评估
- Guardrails AI: 可配置阈值与动作映射
- NeMo Guardrails: 分层 Rail + on_fail 修正
- SelfCheckGPT: 多次采样自洽性检查
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .layers import (
    BaseReviewLayer,
    FactLayer,
    LayerResult,
    LayerRuleResult,
    LogicLayer,
    NumericalLayer,
    ProvenanceLayer,
    ReviewLayerType,
    RuleSeverity,
)
from .models import Claim, Evidence, EvidenceType, VerificationRequest
from .pipeline import ClaimExtractor, EvidenceCollector
from .scoring import CompositeScoringEngine
from .state_machine import (
    ReviewState,
    ReviewStateMachine,
    ReviewVerdict,
    SelfCorrectionLoop,
)


# ============================================================
# 学习者水平枚举 (动态阈值)
# ============================================================


class LearnerLevel(str, Enum):
    """学习者水平 — 驱动动态阈值调整.

    不同水平的学习者对内容精度的要求不同:
    - 初学者: 事实层阈值降低, 逻辑/数值/溯源层关闭
    - 进阶者: 事实层阈值提高, 逻辑/数值层开启, 溯源层关闭
    - 专家: 全层开启, 标准阈值
    - 教师: 全层开启, 严格阈值
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    TEACHER = "teacher"


#: 学习者水平 → 阈值映射
LEARNER_LEVEL_THRESHOLDS: dict[LearnerLevel, dict[str, float]] = {
    LearnerLevel.BEGINNER: {
        "pass_threshold": 70.0,
        "flag_threshold": 45.0,
        "enable_l2": 0.0,  # 关闭
        "enable_l3": 0.0,
        "enable_l4": 0.0,
    },
    LearnerLevel.INTERMEDIATE: {
        "pass_threshold": 80.0,
        "flag_threshold": 55.0,
        "enable_l2": 1.0,  # 开启
        "enable_l3": 1.0,
        "enable_l4": 0.0,  # 关闭
    },
    LearnerLevel.EXPERT: {
        "pass_threshold": 85.0,
        "flag_threshold": 60.0,
        "enable_l2": 1.0,
        "enable_l3": 1.0,
        "enable_l4": 1.0,
    },
    LearnerLevel.TEACHER: {
        "pass_threshold": 90.0,
        "flag_threshold": 70.0,
        "enable_l2": 1.0,
        "enable_l3": 1.0,
        "enable_l4": 1.0,
    },
}


# ============================================================
# 管道配置
# ============================================================


@dataclass
class ReviewPipelineConfig:
    """评审管道配置.

    Attributes:
        pass_threshold: 通过阈值 (默认 85.0)
        flag_threshold: 警告阈值 (默认 60.0)
        max_corrections: 最大自纠次数 (默认 2)
        enable_self_correction: 是否启用自纠回路 (默认 True)
        enable_short_circuit: 是否启用四层递进短路 (默认 True)
            — 任一层 BLOCK 则跳过后续层, 直接进入评分
        learner_level: 学习者水平 (默认 EXPERT)
            — 驱动动态阈值调整, 覆盖 pass/flag_threshold
        error_penalty: ERROR 级别失败扣分 (默认 30.0)
        critical_penalty: CRITICAL 级别失败扣分 (默认 50.0)
        warning_penalty: WARNING 级别失败扣分 (默认 10.0)
        info_penalty: INFO 级别失败扣分 (默认 5.0)
    """

    pass_threshold: float = 85.0
    flag_threshold: float = 60.0
    max_corrections: int = 2
    enable_self_correction: bool = True
    enable_short_circuit: bool = True
    learner_level: LearnerLevel = LearnerLevel.EXPERT
    error_penalty: float = 30.0
    critical_penalty: float = 50.0
    warning_penalty: float = 10.0
    info_penalty: float = 5.0

    def __post_init__(self) -> None:
        """根据学习者水平动态调整阈值.

        仅当学习者水平非默认值 (EXPERT) 时覆盖阈值,
        以便用户自定义的阈值在默认水平下仍然生效.
        """
        if self.learner_level != LearnerLevel.EXPERT:
            level_config = LEARNER_LEVEL_THRESHOLDS.get(self.learner_level)
            if level_config:
                self.pass_threshold = level_config["pass_threshold"]
                self.flag_threshold = level_config["flag_threshold"]


# ============================================================
# 评审结果
# ============================================================


@dataclass
class ReviewResult:
    """评审结果.

    Attributes:
        report_id: 报告 ID
        request_id: 请求 ID
        agent_id: Agent ID
        original_output: 原始输出
        verdict: 评审判决 (PASS / FLAG / BLOCK)
        composite_score: 综合评分 (0-100)
        layer_results: 各层评审结果
        layer_scores: 各层评分
        self_correction: 自纠回路 (如果触发)
        issues: 问题列表
        corrected_output: 修正后的输出 (如果有)
        created_at: 创建时间
        completed_at: 完成时间
        metadata: 附加元数据
    """

    report_id: str = field(
        default_factory=lambda: f"rr-{uuid.uuid4().hex[:12]}"
    )
    request_id: str = ""
    agent_id: str = ""
    original_output: str = ""
    verdict: ReviewVerdict = ReviewVerdict.PASS
    composite_score: float = 0.0
    layer_results: dict[ReviewLayerType, LayerResult] = field(
        default_factory=dict
    )
    layer_scores: dict[ReviewLayerType, float] = field(
        default_factory=dict
    )
    self_correction: SelfCorrectionLoop | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    corrected_output: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 评审管道
# ============================================================


class ReviewPipeline:
    """增强评审管道.

    编排四层评审 + 状态机 + 自纠回路 + 综合评分.

    使用示例::

        pipeline = ReviewPipeline()
        request = VerificationRequest(
            agent_id="agent-knowledge",
            output_text="Dy3+ 的发射主峰在 575nm。",
            context_chunks=["Dy3+ 发射峰 575nm"],
        )
        result = pipeline.review(request)
        assert result.verdict == ReviewVerdict.PASS
    """

    def __init__(
        self,
        config: ReviewPipelineConfig | None = None,
        fact_layer: FactLayer | None = None,
        logic_layer: LogicLayer | None = None,
        numerical_layer: NumericalLayer | None = None,
        provenance_layer: ProvenanceLayer | None = None,
        scoring_engine: CompositeScoringEngine | None = None,
    ) -> None:
        self._config = config or ReviewPipelineConfig()
        self._fact_layer = fact_layer or FactLayer()
        self._logic_layer = logic_layer or LogicLayer()
        self._numerical_layer = numerical_layer or NumericalLayer()
        self._provenance_layer = provenance_layer or ProvenanceLayer()
        # 使用配置中的动态阈值初始化评分引擎
        self._scoring_engine = scoring_engine or CompositeScoringEngine(
            pass_threshold=self._config.pass_threshold,
            flag_threshold=self._config.flag_threshold,
        )
        self._state_machine = ReviewStateMachine()
        self._extractor = ClaimExtractor()
        self._collector = EvidenceCollector()
        self._report_store: dict[str, ReviewResult] = {}

    # --------------------------------------------------------
    # 属性访问
    # --------------------------------------------------------

    @property
    def config(self) -> ReviewPipelineConfig:
        return self._config

    @property
    def fact_layer(self) -> FactLayer:
        return self._fact_layer

    @property
    def logic_layer(self) -> LogicLayer:
        return self._logic_layer

    @property
    def numerical_layer(self) -> NumericalLayer:
        return self._numerical_layer

    @property
    def provenance_layer(self) -> ProvenanceLayer:
        return self._provenance_layer

    @property
    def scoring_engine(self) -> CompositeScoringEngine:
        return self._scoring_engine

    @property
    def state_machine(self) -> ReviewStateMachine:
        return self._state_machine

    @property
    def report_store(self) -> dict[str, "ReviewResult"]:
        """评审报告存储."""
        return self._report_store

    def store_result(self, result: "ReviewResult") -> str:
        """存储评审结果, 返回 report_id."""
        self._report_store[result.report_id] = result
        return result.report_id

    def get_result(self, report_id: str) -> "ReviewResult | None":
        """根据 report_id 获取评审结果."""
        return self._report_store.get(report_id)

    def list_results(
        self,
        agent_id: str | None = None,
        verdict: "ReviewVerdict | None" = None,
        limit: int = 50,
    ) -> list["ReviewResult"]:
        """列出评审结果, 支持过滤."""
        results = list(self._report_store.values())
        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        if verdict:
            results = [r for r in results if r.verdict == verdict]
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """获取评审统计信息."""
        total = len(self._report_store)
        if total == 0:
            return {
                "total": 0,
                "pass": 0,
                "flag": 0,
                "block": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
            }
        pass_count = sum(
            1 for r in self._report_store.values()
            if r.verdict == ReviewVerdict.PASS
        )
        flag_count = sum(
            1 for r in self._report_store.values()
            if r.verdict == ReviewVerdict.FLAG
        )
        block_count = sum(
            1 for r in self._report_store.values()
            if r.verdict == ReviewVerdict.BLOCK
        )
        avg_score = sum(
            r.composite_score for r in self._report_store.values()
        ) / total
        return {
            "total": total,
            "pass": pass_count,
            "flag": flag_count,
            "block": block_count,
            "pass_rate": round(pass_count / total * 100, 1),
            "avg_score": round(avg_score, 2),
        }

    # --------------------------------------------------------
    # 核心评审方法
    # --------------------------------------------------------

    def review(self, request: VerificationRequest) -> ReviewResult:
        """执行完整评审.

        Args:
            request: 验证请求

        Returns:
            评审结果
        """
        result = ReviewResult(
            request_id=request.request_id,
            agent_id=request.agent_id,
            original_output=request.output_text,
            created_at=time.time(),
        )

        # 重置状态机
        self._state_machine.reset()

        try:
            # Stage 1: 声明提取
            claims = self._extractor.extract(request.output_text)
            if not claims:
                result.verdict = ReviewVerdict.PASS
                result.composite_score = 100.0
                result.completed_at = time.time()
                return result

            # Stage 2: 证据收集
            evidence = self._collector.collect(request, claims)

            # Stage 3: 四层评审 (递进短路)
            context_chunks = request.context_chunks
            level_config = LEARNER_LEVEL_THRESHOLDS.get(
                self._config.learner_level, {}
            )
            short_circuit = self._config.enable_short_circuit
            blocked = False

            # L1 事实层 (始终执行)
            self._state_machine.transition(ReviewState.L1_FACT)
            l1_result = self._fact_layer.verify_claims(
                claims,
                context_chunks=context_chunks,
                evidence=evidence,
            )
            result.layer_results[ReviewLayerType.L1_FACT] = l1_result
            l1_verdict = self._layer_verdict(l1_result)
            self._state_machine.transition(
                ReviewState.L2_LOGIC,
                verdict=l1_verdict,
            )

            if short_circuit and l1_verdict == ReviewVerdict.BLOCK:
                blocked = True

            # L2 逻辑层 (根据学习者水平决定是否执行)
            if not blocked and level_config.get("enable_l2", 1.0) > 0:
                l2_result = self._logic_layer.verify_claims(
                    claims,
                    context_chunks=context_chunks,
                    evidence=evidence,
                )
                result.layer_results[ReviewLayerType.L2_LOGIC] = l2_result
                l2_verdict = self._layer_verdict(l2_result)
                self._state_machine.transition(
                    ReviewState.L3_NUMERICAL,
                    verdict=l2_verdict,
                )
                if short_circuit and l2_verdict == ReviewVerdict.BLOCK:
                    blocked = True
            else:
                self._state_machine.transition(
                    ReviewState.L3_NUMERICAL,
                    verdict=ReviewVerdict.PASS,
                )

            # L3 数值层
            if not blocked and level_config.get("enable_l3", 1.0) > 0:
                l3_result = self._numerical_layer.verify_claims(
                    claims,
                    context_chunks=context_chunks,
                    evidence=evidence,
                )
                result.layer_results[ReviewLayerType.L3_NUMERICAL] = l3_result
                l3_verdict = self._layer_verdict(l3_result)
                self._state_machine.transition(
                    ReviewState.L4_PROVENANCE,
                    verdict=l3_verdict,
                )
                if short_circuit and l3_verdict == ReviewVerdict.BLOCK:
                    blocked = True
            else:
                self._state_machine.transition(
                    ReviewState.L4_PROVENANCE,
                    verdict=ReviewVerdict.PASS,
                )

            # L4 溯源层
            if not blocked and level_config.get("enable_l4", 1.0) > 0:
                l4_result = self._provenance_layer.verify_claims(
                    claims,
                    context_chunks=context_chunks,
                    evidence=evidence,
                )
                result.layer_results[ReviewLayerType.L4_PROVENANCE] = l4_result
                l4_verdict = self._layer_verdict(l4_result)
                self._state_machine.transition(
                    ReviewState.COMPOSITE_SCORE,
                    verdict=l4_verdict,
                )
            else:
                self._state_machine.transition(
                    ReviewState.COMPOSITE_SCORE,
                    verdict=ReviewVerdict.PASS,
                )

            # Stage 4: 综合评分 (惩罚制)
            layer_scores = self._compute_layer_scores(result.layer_results)
            result.layer_scores = layer_scores

            composite = self._scoring_engine.compute_score(
                l1=layer_scores[ReviewLayerType.L1_FACT],
                l2=layer_scores[ReviewLayerType.L2_LOGIC],
                l3=layer_scores[ReviewLayerType.L3_NUMERICAL],
                l4=layer_scores[ReviewLayerType.L4_PROVENANCE],
            )
            result.composite_score = composite
            verdict = self._scoring_engine.determine_verdict(composite)

            # 收集问题
            result.issues = self._collect_issues(result.layer_results)

            # Stage 5: 自纠回路 (如果 FLAG 或 BLOCK)
            if (
                self._config.enable_self_correction
                and verdict in (ReviewVerdict.FLAG, ReviewVerdict.BLOCK)
            ):
                self._state_machine.transition(
                    ReviewState.SELF_CORRECT,
                    verdict=verdict,
                    score=composite,
                )
                result.self_correction = self._run_self_correction(
                    request=request,
                    claims=claims,
                    evidence=evidence,
                    context_chunks=context_chunks,
                    initial_verdict=verdict,
                )

                # 自纠后更新判决
                if result.self_correction.is_resolved:
                    verdict = ReviewVerdict.PASS
                    result.corrected_output = (
                        result.self_correction.get_latest_suggestions()[0]
                        if result.self_correction.get_latest_suggestions()
                        else None
                    )
                elif result.self_correction.needs_escalation:
                    verdict = ReviewVerdict.BLOCK

            # 设置最终判决
            result.verdict = verdict

            # 状态机转到终态
            if verdict == ReviewVerdict.PASS:
                self._state_machine.transition(
                    ReviewState.PASS,
                    verdict=verdict,
                    score=composite,
                )
            elif verdict == ReviewVerdict.FLAG:
                self._state_machine.transition(
                    ReviewState.FLAG,
                    verdict=verdict,
                    score=composite,
                )
            else:
                self._state_machine.transition(
                    ReviewState.BLOCK,
                    verdict=verdict,
                    score=composite,
                )

        except Exception as exc:
            result.verdict = ReviewVerdict.BLOCK
            result.metadata["error"] = str(exc)
            result.composite_score = 0.0

        result.completed_at = time.time()
        # 自动存储评审结果
        self.store_result(result)
        return result

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    @staticmethod
    def _layer_verdict(layer_result: LayerResult) -> ReviewVerdict:
        """将层 verdict 字符串转为 ReviewVerdict 枚举."""
        v = layer_result.verdict.upper()
        if v == "PASS":
            return ReviewVerdict.PASS
        elif v == "BLOCK":
            return ReviewVerdict.BLOCK
        else:
            return ReviewVerdict.FLAG

    def _compute_layer_scores(
        self,
        layer_results: dict[ReviewLayerType, LayerResult],
    ) -> dict[ReviewLayerType, float]:
        """计算各层评分 (惩罚制).

        基础分 100, 每条失败规则按严重级别扣分.
        """
        scores: dict[ReviewLayerType, float] = {}

        penalty_map = {
            RuleSeverity.CRITICAL: self._config.critical_penalty,
            RuleSeverity.ERROR: self._config.error_penalty,
            RuleSeverity.WARNING: self._config.warning_penalty,
            RuleSeverity.INFO: self._config.info_penalty,
        }

        for layer_type, layer_result in layer_results.items():
            penalty = 0.0
            for rule_result in layer_result.rule_results:
                if not rule_result.passed:
                    penalty += penalty_map.get(rule_result.severity, 10.0)
            score = max(0.0, 100.0 - penalty)
            scores[layer_type] = round(score, 2)

        return scores

    @staticmethod
    def _collect_issues(
        layer_results: dict[ReviewLayerType, LayerResult],
    ) -> list[dict[str, Any]]:
        """收集所有失败规则作为问题列表."""
        issues: list[dict[str, Any]] = []
        for layer_type, layer_result in layer_results.items():
            for rule_result in layer_result.rule_results:
                if not rule_result.passed:
                    issues.append({
                        "rule_id": rule_result.rule_id,
                        "rule": rule_result.rule_name,
                        "layer": layer_type.value,
                        "severity": rule_result.severity.value,
                        "detail": rule_result.detail,
                    })
        return issues

    def _run_self_correction(
        self,
        request: VerificationRequest,
        claims: list[Claim],
        evidence: list[Evidence],
        context_chunks: list[str],
        initial_verdict: ReviewVerdict,
    ) -> SelfCorrectionLoop:
        """执行自纠回路.

        策略:
        1. 收集问题列表
        2. 生成修正建议
        3. 尝试修正输出
        4. 重新评审
        5. 如果通过则结束, 否则继续重试
        """
        loop = SelfCorrectionLoop(max_attempts=self._config.max_corrections)

        # 收集初始问题
        issues = self._collect_issues_from_claims(
            claims, evidence, context_chunks
        )
        suggestions = self._generate_suggestions(issues)

        while loop.can_retry and not loop.is_resolved:
            record = loop.start_correction(
                issues=issues,
                suggestions=suggestions,
            )

            # 尝试修正输出
            corrected_text = self._apply_corrections(
                request.output_text,
                issues,
            )

            # 重新评审修正后的输出
            corrected_request = VerificationRequest(
                agent_id=request.agent_id,
                output_text=corrected_text,
                context_chunks=context_chunks,  # 上下文不变
                citations=request.citations,
                sample_outputs=request.sample_outputs,
                reference_answer=request.reference_answer,
            )

            corrected_claims = self._extractor.extract(corrected_text)
            corrected_evidence = self._collector.collect(
                corrected_request, corrected_claims
            )

            # 重新四层评审
            l1 = self._fact_layer.verify_claims(
                corrected_claims,
                context_chunks=context_chunks,
                evidence=corrected_evidence,
            )
            l2 = self._logic_layer.verify_claims(
                corrected_claims,
                context_chunks=context_chunks,
                evidence=corrected_evidence,
            )
            l3 = self._numerical_layer.verify_claims(
                corrected_claims,
                context_chunks=context_chunks,
                evidence=corrected_evidence,
            )
            l4 = self._provenance_layer.verify_claims(
                corrected_claims,
                context_chunks=context_chunks,
                evidence=corrected_evidence,
            )

            layer_scores = self._compute_layer_scores({
                ReviewLayerType.L1_FACT: l1,
                ReviewLayerType.L2_LOGIC: l2,
                ReviewLayerType.L3_NUMERICAL: l3,
                ReviewLayerType.L4_PROVENANCE: l4,
            })

            composite = self._scoring_engine.compute_score(
                l1=layer_scores[ReviewLayerType.L1_FACT],
                l2=layer_scores[ReviewLayerType.L2_LOGIC],
                l3=layer_scores[ReviewLayerType.L3_NUMERICAL],
                l4=layer_scores[ReviewLayerType.L4_PROVENANCE],
            )

            corrected_verdict = self._scoring_engine.determine_verdict(
                composite
            )

            success = corrected_verdict == ReviewVerdict.PASS
            loop.record_result(
                record.attempt_number,
                success=success,
                corrected_output=corrected_text if success else None,
            )

            if not success:
                # 更新问题列表用于下次修正
                issues = self._collect_issues_from_claims(
                    corrected_claims, corrected_evidence, context_chunks
                )
                suggestions = self._generate_suggestions(issues)

        return loop

    def _collect_issues_from_claims(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
        context_chunks: list[str],
    ) -> list[str]:
        """从声明评审中收集问题."""
        issues: list[str] = []

        l1 = self._fact_layer.verify_claims(
            claims,
            context_chunks=context_chunks,
            evidence=evidence,
        )
        l2 = self._logic_layer.verify_claims(
            claims,
            context_chunks=context_chunks,
            evidence=evidence,
        )
        l3 = self._numerical_layer.verify_claims(
            claims,
            context_chunks=context_chunks,
            evidence=evidence,
        )
        l4 = self._provenance_layer.verify_claims(
            claims,
            context_chunks=context_chunks,
            evidence=evidence,
        )

        for layer_result in [l1, l2, l3, l4]:
            for rule_result in layer_result.rule_results:
                if not rule_result.passed:
                    issues.append(
                        f"{rule_result.rule_id} {rule_result.rule_name}: "
                        f"{rule_result.detail}"
                    )

        return issues

    @staticmethod
    def _generate_suggestions(issues: list[str]) -> list[str]:
        """根据问题生成修正建议.

        融合 CoVe 策略: 基于失败规则类型生成针对性修正方向.
        """
        suggestions: list[str] = []
        for issue in issues:
            if "发射峰" in issue and ("超出" in issue or "不在" in issue):
                suggestions.append(
                    "将发射峰波长修正为 570-585nm 范围内的值 (如 575nm)"
                )
            elif "蓝色发射" in issue and ("超出" in issue or "不在" in issue):
                suggestions.append(
                    "将蓝色发射峰波长修正为 470-490nm 范围内的值 (如 480nm)"
                )
            elif "激发" in issue and ("超出" in issue or "不在" in issue):
                suggestions.append(
                    "将激发波长修正为 350-460nm 范围内的值 (如 350nm 或 450nm)"
                )
            elif "浓度" in issue and ("超出" in issue or "低于" in issue):
                suggestions.append(
                    "将掺杂浓度修正为 1-5mol% 范围内的值 (如 2mol%)"
                )
            elif "d 区" in issue or "过渡金属" in issue:
                suggestions.append(
                    "将分类修正为: Dy3+ 属于镧系元素, f 区"
                )
            elif "寿命" in issue and "超出" in issue:
                suggestions.append(
                    "将衰减寿命修正为 0.1-2.0ms 范围内的值"
                )
            elif "量子效率" in issue:
                suggestions.append(
                    "将量子效率修正为 0-100% 范围内的值"
                )
            elif "CIE" in issue or "色坐标" in issue:
                suggestions.append(
                    "将 CIE 色坐标修正为 x: 0-1, y: 0-1 范围内的值"
                )
            elif "色温" in issue:
                suggestions.append(
                    "将色温修正为 3000-8000K 范围内的值"
                )
            elif "温度" in issue and "升高" in issue:
                suggestions.append(
                    "热猝灭效应: 温度升高时发光强度应降低, 修正描述"
                )
            elif "能级" in issue or "跃迁" in issue:
                suggestions.append(
                    "修正能级跃迁: 黄色发射=⁴F₉/₂→⁶H₁₃/₂, 蓝色发射=⁴F₉/₂→⁶H₁₅/₂"
                )
            elif "AI" in issue or "来源" in issue:
                suggestions.append(
                    "为声明添加来源引用或标注为 AI-generated"
                )
            else:
                suggestions.append(f"修正问题: {issue}")
        return suggestions

    @staticmethod
    def _apply_corrections(
        output_text: str,
        issues: list[str],
    ) -> str:
        """应用修正到输出文本.

        策略 (CoVe 式自动修正):
        - 替换超出范围的发射峰波长为 575nm
        - 替换超出范围的蓝色发射峰为 480nm
        - 替换超出范围的激发波长为 350nm
        - 替换超出范围的浓度为 2mol%
        - 替换超出范围的寿命为 1.0ms
        - 替换超出范围的量子效率为 50%
        - 替换错误分类为镧系
        """
        corrected = output_text

        # 替换发射峰波长 (查找所有 nm 值, 替换超出范围的)
        nm_pattern = re.compile(r"(\d+\.?\d*)\s*nm", re.IGNORECASE)
        nm_matches = nm_pattern.findall(corrected)
        for match in nm_matches:
            val = float(match)
            if val < 570 or val > 585:
                # 判断是发射峰还是激发波长
                context_start = max(0, corrected.find(match) - 10)
                context = corrected[context_start:context_start + 20]
                if "激发" in context or "excitation" in context.lower():
                    replacement = "350nm"
                elif "蓝" in context or "blue" in context.lower():
                    replacement = "480nm"
                else:
                    replacement = "575nm"
                corrected = corrected.replace(f"{match}nm", replacement)
                corrected = corrected.replace(f"{match} nm", replacement)

        # 替换浓度
        conc_pattern = re.compile(r"(\d+\.?\d*)\s*mol%", re.IGNORECASE)
        conc_matches = conc_pattern.findall(corrected)
        for match in conc_matches:
            val = float(match)
            if val < 1 or val > 5:
                corrected = corrected.replace(f"{match}mol%", "2mol%")
                corrected = corrected.replace(f"{match} mol%", "2mol%")

        # 替换衰减寿命
        ms_pattern = re.compile(r"(\d+\.?\d*)\s*ms", re.IGNORECASE)
        ms_matches = ms_pattern.findall(corrected)
        for match in ms_matches:
            val = float(match)
            if val < 0.1 or val > 2.0:
                corrected = corrected.replace(f"{match}ms", "1.0ms")
                corrected = corrected.replace(f"{match} ms", "1.0ms")

        # 替换量子效率
        qe_pattern = re.compile(r"(\d+\.?\d*)\s*%", re.IGNORECASE)
        qe_matches = qe_pattern.findall(corrected)
        for match in qe_matches:
            val = float(match)
            if val > 100:
                corrected = corrected.replace(f"{match}%", "50%")

        # 替换错误分类
        corrected = corrected.replace("d 区", "镧系")
        corrected = corrected.replace("d区", "镧系")
        corrected = corrected.replace("d-block", "f-block")
        corrected = corrected.replace("过渡金属", "稀土元素")

        return corrected
