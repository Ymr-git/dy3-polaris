"""L4 决策引擎层 — 增强版验证与策略评估编排器 (ValidationOrchestrator).

融合世界先进方案的多维度验证设计:
- PRISM MHCV (2025): 多维度异构协同验证
  - 多维度: 事实性/质量/一致性/时效性/可信度/相关性
  - 异构: 不同验证器使用不同算法 (规则/统计/图/嵌入)
  - 协同: 验证器间结果互相印证
  - 聚合: Discard-Weighted Voting (丢弃低质量后加权投票)
- FActScore + ProVe + SAFE: 原子化事实抽取与校验
- OQuaRE-KG: 六维质量评估框架
- MACR + CRDL: 多智能体冲突检测与消解
- RAGAS (2024): RAG 系统质量评估 — Faithfulness/Relevancy/Precision/Recall
- UniCR (2025): 不确定性量化网关 — 多源信号融合 + 分层验证
- CISC (2025): 置信度感知自洽性 — 加权多数投票
- VeriCoT (2025): 神经符号 CoT 验证 — 逻辑一致性检查
- VeReaFine (2025): 迭代验证-推理-精炼 — 缺失证据识别
- VRR-Stop (2026): 验证-修复循环的鲁棒停止框架
- PRISM (2026): 增益分解理论 — 探索/信息/聚合三维策略评估
- HydraRAG (2025): 三因子评分 — 多源交叉验证

核心职责:
    对 T3(ExecutionResult) 执行 UQ 驱动的分层多维度验证，产出增强版 ValidationReport。
    串联 L3 层已有的 FactChecker、QualityManager、ConflictDetector。

验证维度:
    1. 事实校验 — 数值声明与标准值比对
    2. 质量评估 — 六维质量评分
    3. 冲突检测 — 知识冲突识别
    4. 合规检查 — 策略与约束合规
    5. Faithfulness 评估 — 生成答案与检索上下文的事实一致性
    6. 自洽性检查 — 多路径推理答案的一致性
    7. 策略评估 — 推理策略优劣评估与优化建议

验证层级:
    L1 轻量验证 — 高置信度: 基础四维度 + Faithfulness 快速扫描
    L2 标准验证 — 中等置信度: L1 + 完整 Faithfulness + 自洽性 + 多文献交叉
    L3 深度验证 — 低置信度: L2 + 策略评估 + 深度逻辑验证
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .domain_rule_engine import DomainRuleEngine
from .faithfulness_checker import FaithfulnessChecker, SelfConsistencyChecker
from .models import (
    DecisionPlan,
    ExecutionResult,
    ExecutionStatus,
    TaskResult,
    TaskType,
    ValidationReport,
    ValidationSeverity,
    ValidationTier,
)
from .strategy_evaluator import StrategyEvaluator
from .uq_gate import UQAssessment, UQGate
from .vr_loop import VRLoopController

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """增强版验证编排器 — T4 核心模块.

    整合 UQ 驱动的分层验证 + 多维度异构协同验证 + V&R 闭环。

    借鉴 PRISM MHCV 的多维度异构协同验证架构:
    - 并行执行多个验证器
    - 丢弃低置信度结果 (Discard)
    - 对剩余结果加权投票 (Weighted Voting)
    - 生成统一验证报告

    Usage::

        orchestrator = ValidationOrchestrator(
            fact_checker=fact_checker,
            quality_manager=quality_manager,
        )
        report = orchestrator.validate(execution_result)
        # report 被传入 ActionSelector
    """

    def __init__(
        self,
        fact_checker: Any | None = None,
        quality_manager: Any | None = None,
        conflict_detector: Any | None = None,
        *,
        # 基础维度权重
        fact_check_weight: float = 0.22,
        quality_weight: float = 0.18,
        conflict_weight: float = 0.13,
        compliance_weight: float = 0.09,
        # 增强维度权重
        faithfulness_weight: float = 0.13,
        self_consistency_weight: float = 0.09,
        strategy_eval_weight: float = 0.05,
        domain_rule_weight: float = 0.11,
        # 聚合参数
        discard_threshold: float = 0.3,
        # UQ 网关参数
        uq_l1_threshold: float = 0.75,
        uq_l2_threshold: float = 0.50,
        uq_l3_threshold: float = 0.30,
        # V&R 闭环参数
        enable_vr_loop: bool = True,
        vr_max_iterations: int = 3,
        vr_min_improvement: float = 0.05,
        # 领域规则引擎参数
        enable_domain_rules: bool = True,
        domain_rule_engine: DomainRuleEngine | None = None,
    ) -> None:
        """初始化增强版验证编排器.

        Args:
            fact_checker: 事实校验器 (FactChecker)
            quality_manager: 质量管理器 (QualityManager)
            conflict_detector: 冲突检测器 (ConflictDetector)
            fact_check_weight: 事实校验权重
            quality_weight: 质量评估权重
            conflict_weight: 冲突检测权重
            compliance_weight: 合规检查权重
            faithfulness_weight: Faithfulness 评估权重
            self_consistency_weight: 自洽性检查权重
            strategy_eval_weight: 策略评估权重
            domain_rule_weight: 领域规则权重
            discard_threshold: 丢弃阈值 (低于此分数的结果不参与聚合)
            uq_l1_threshold: UQ L1 阈值
            uq_l2_threshold: UQ L2 阈值
            uq_l3_threshold: UQ L3 阈值
            enable_vr_loop: 是否启用 V&R 闭环
            vr_max_iterations: V&R 最大迭代轮次
            vr_min_improvement: V&R 最小改善幅度
            enable_domain_rules: 是否启用领域规则引擎
            domain_rule_engine: 自定义领域规则引擎实例 (None 则自动创建)
        """
        self._fact_checker = fact_checker
        self._quality_manager = quality_manager
        self._conflict_detector = conflict_detector

        self._weights = {
            "fact_check": fact_check_weight,
            "quality": quality_weight,
            "conflict": conflict_weight,
            "compliance": compliance_weight,
            "faithfulness": faithfulness_weight,
            "self_consistency": self_consistency_weight,
            "strategy_eval": strategy_eval_weight,
            "domain_rules": domain_rule_weight,
        }
        self._discard_threshold = discard_threshold

        # 增强模块
        self._uq_gate = UQGate(
            l1_threshold=uq_l1_threshold,
            l2_threshold=uq_l2_threshold,
            l3_threshold=uq_l3_threshold,
        )
        self._faithfulness_checker = FaithfulnessChecker()
        self._self_consistency_checker = SelfConsistencyChecker()
        self._strategy_evaluator = StrategyEvaluator()

        # 领域规则引擎
        self._enable_domain_rules = enable_domain_rules
        if enable_domain_rules:
            self._domain_rule_engine: DomainRuleEngine | None = (
                domain_rule_engine if domain_rule_engine is not None else DomainRuleEngine()
            )
        else:
            self._domain_rule_engine = None

        # V&R 闭环
        self._enable_vr_loop = enable_vr_loop
        self._vr_controller: VRLoopController | None = None
        if enable_vr_loop:
            self._vr_controller = VRLoopController(
                max_iterations=vr_max_iterations,
                min_improvement=vr_min_improvement,
            )

        logger.info(
            "增强版 ValidationOrchestrator 初始化完成 "
            "(权重: fact=%.2f, quality=%.2f, conflict=%.2f, compliance=%.2f, "
            "faithfulness=%.2f, consistency=%.2f, strategy=%.2f, domain=%.2f, "
            "vr_loop=%s, domain_rules=%s)",
            fact_check_weight, quality_weight, conflict_weight, compliance_weight,
            faithfulness_weight, self_consistency_weight, strategy_eval_weight,
            domain_rule_weight, enable_vr_loop, enable_domain_rules,
        )

    def validate(
        self,
        execution_result: ExecutionResult,
        *,
        plan: DecisionPlan | None = None,
        intent_type: str = "",
        historical_feedback: dict[str, float] | None = None,
    ) -> ValidationReport:
        """对执行结果执行增强版多维度验证.

        Args:
            execution_result: T3 产出的执行结果
            plan: T2 生成的决策计划 (可选，用于策略评估)
            intent_type: 意图类型 (用于 UQ 网关先验)
            historical_feedback: 历史反馈统计

        Returns:
            ValidationReport 增强版验证报告
        """
        start_ts = time.perf_counter()
        plan_id = execution_result.plan_id

        logger.info("开始验证执行结果: plan_id=%s", plan_id)

        report = ValidationReport(plan_id=plan_id)

        # 若执行失败，直接标记为 ERROR
        if execution_result.status == ExecutionStatus.FAILED:
            report.overall_status = ValidationSeverity.ERROR
            report.overall_score = 0.0
            report.anomalies.append({
                "type": "execution_failure",
                "message": execution_result.error_summary or "执行失败",
                "severity": "error",
            })
            report.validation_time_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            return report

        # Step 1: UQ 驱动的验证层级选择
        uq_result = self._uq_gate.assess(
            execution_result,
            intent_type=intent_type,
            historical_feedback=historical_feedback,
        )
        report.validation_tier = uq_result.tier
        report.uq_score = uq_result.score

        logger.info(
            "UQ 评估: score=%.4f, tier=%s",
            uq_result.score, uq_result.tier.value,
        )

        # Step 2: 基础四维度验证 (所有层级都执行)
        fact_result = self._run_fact_check(execution_result)
        quality_result = self._run_quality_assessment(execution_result)
        conflict_result = self._run_conflict_detection(execution_result)
        compliance_result = self._run_compliance_check(execution_result)

        report.fact_check = fact_result
        report.quality_assessment = quality_result
        report.conflict_detection = conflict_result
        report.compliance_check = compliance_result

        # Step 3: L2+ 增强验证 — Faithfulness + 自洽性
        if uq_result.tier.value != "l1_lightweight":
            faithfulness_result = self._faithfulness_checker.assess(execution_result)
            consistency_result = self._self_consistency_checker.assess(execution_result)

            report.faithfulness_assessment = faithfulness_result
            report.self_consistency = consistency_result
        else:
            # L1 轻量验证: 快速 Faithfulness 扫描
            faithfulness_result = self._faithfulness_checker.assess(execution_result)
            report.faithfulness_assessment = {
                "faithfulness_score": faithfulness_result.get("faithfulness_score", 1.0),
                "total_claims": faithfulness_result.get("total_claims", 0),
                "supported_claims": faithfulness_result.get("supported_claims", 0),
                "quick_scan": True,
            }
            report.self_consistency = {}

        # Step 4: L3 深度验证 — 策略评估
        if uq_result.tier.value == "l3_deep":
            strategy_result = self._strategy_evaluator.evaluate(plan, execution_result)
            report.strategy_evaluation = strategy_result
        else:
            report.strategy_evaluation = {}

        # Step 4.5: 领域规则验证 (所有层级都执行，但深度可配置)
        domain_result: dict[str, Any] = {}
        if self._domain_rule_engine is not None:
            try:
                domain_result = self._domain_rule_engine.evaluate(execution_result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("领域规则评估异常")
                domain_result = {
                    "overall_score": 0.5,
                    "error": str(exc),
                    "rule_results": [],
                    "all_violations": [],
                    "high_severity_violations": [],
                }
        report.domain_rule_results = domain_result

        # Step 5: 聚合评分 (Discard-Weighted Voting)
        report.overall_score = self._aggregate_scores(
            fact_result, quality_result, conflict_result, compliance_result,
            faithfulness_result if isinstance(faithfulness_result, dict) else {},
            report.self_consistency,
            report.strategy_evaluation,
            domain_result,
        )
        report.overall_status = self._score_to_status(report.overall_score)

        # Step 6: 收集异常与建议
        report.anomalies = self._collect_anomalies(
            fact_result, quality_result, conflict_result, compliance_result,
            report.faithfulness_assessment, report.self_consistency,
            domain_result,
        )
        report.recommendations = self._generate_recommendations(
            fact_result, quality_result, conflict_result, compliance_result,
            report.faithfulness_assessment, report.self_consistency,
            report.strategy_evaluation, domain_result,
        )

        # Step 7: V&R 闭环信息
        if self._vr_controller is not None:
            # 记录 V&R 控制器状态 (实际修正由上层 DecisionEngine 执行)
            report.refinement_iterations = 0
            report.refinement_history = []

        report.validation_time_ms = round((time.perf_counter() - start_ts) * 1000, 2)

        logger.info(
            "验证完成: plan_id=%s, tier=%s, 总体状态=%s, 综合分数=%.4f, "
            "异常=%d, UQ=%.4f, 领域规则=%s, 耗时=%.2fms",
            plan_id, uq_result.tier.value, report.overall_status.value,
            report.overall_score, len(report.anomalies), uq_result.score,
            "启用" if self._domain_rule_engine else "禁用",
            report.validation_time_ms,
        )

        return report

    # --------------------------------------------------------
    # V&R 闭环支持
    # --------------------------------------------------------

    def get_vr_controller(self) -> VRLoopController | None:
        """获取 V&R 闭环控制器 (供 DecisionEngine 使用)."""
        return self._vr_controller

    def get_domain_rule_engine(self) -> DomainRuleEngine | None:
        """获取领域规则引擎 (供外部查询规则状态)."""
        return self._domain_rule_engine

    def generate_refinement_feedback(
        self,
        report: ValidationReport,
        execution_result: ExecutionResult,
    ) -> list[dict[str, Any]]:
        """生成精炼反馈 (供 DecisionEngine 触发修正).

        Returns:
            反馈列表，每条反馈为字典
        """
        if self._vr_controller is None:
            return []

        feedbacks = self._vr_controller.generate_feedback(report, execution_result)
        return [
            {
                "feedback_type": f.feedback_type,
                "severity": f.severity,
                "location": f.location,
                "description": f.description,
                "suggested_action": f.suggested_action,
                "details": f.details,
            }
            for f in feedbacks
        ]

    # --------------------------------------------------------
    # 基础四维度验证 (保持原有实现)
    # --------------------------------------------------------

    def _run_fact_check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """事实校验 — 提取数值声明并与标准值比对."""
        result: dict[str, Any] = {
            "enabled": self._fact_checker is not None,
            "score": 1.0,
            "passed": True,
            "details": [],
        }

        if self._fact_checker is None:
            result["score"] = 1.0  # 无校验器时默认通过
            return result

        try:
            content = self._extract_verifiable_content(execution_result)
            if not content:
                result["message"] = "无可校验内容"
                return result

            check_report = self._fact_checker.check(content)

            result["score"] = check_report.confidence if hasattr(check_report, "confidence") else check_report.pass_rate
            result["passed"] = getattr(check_report, "overall_passed", True)
            result["assertions_checked"] = getattr(check_report, "checked", 0)
            result["assertions_passed"] = getattr(check_report, "passed", 0)
            result["assertions_failed"] = getattr(check_report, "failed", 0)

            if hasattr(check_report, "results"):
                for r in check_report.results:
                    detail = {
                        "text": r.get("text", ""),
                        "status": r.get("status", "skipped"),
                        "deviation": r.get("deviation", 0.0),
                        "message": r.get("message", ""),
                    }
                    result["details"].append(detail)

        except Exception as exc:  # noqa: BLE001
            logger.exception("事实校验异常")
            result["score"] = 0.5
            result["passed"] = False
            result["error"] = str(exc)

        return result

    def _run_quality_assessment(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """质量评估 — 六维质量评分."""
        result: dict[str, Any] = {
            "enabled": self._quality_manager is not None,
            "score": 1.0,
            "dimensions": {},
            "details": [],
        }

        try:
            dim_scores: dict[str, float] = {}

            # 准确性: 基于事实校验和推理置信度
            reason_results = execution_result.get_results_by_type(TaskType.REASON)
            avg_reason_conf = (
                sum(r.confidence for r in reason_results) / len(reason_results)
                if reason_results else 1.0
            )
            dim_scores["accuracy"] = avg_reason_conf

            # 一致性: 检查任务间矛盾
            retrieve_results = execution_result.get_results_by_type(TaskType.RETRIEVE)
            verify_results = execution_result.get_results_by_type(TaskType.VERIFY)
            consistency = 1.0
            if verify_results:
                failed_verify = sum(1 for r in verify_results if r.is_failed)
                consistency = 1.0 - (failed_verify / len(verify_results)) * 0.5
            dim_scores["consistency"] = consistency

            # 完整性: 基于证据数量
            evidence_count = len(execution_result.evidence_set)
            dim_scores["completeness"] = min(1.0, evidence_count / 5.0)

            # 时效性: 默认高分
            dim_scores["timeliness"] = 0.95

            # 可信度: 基于综合置信度
            dim_scores["trustworthiness"] = execution_result.confidence

            # 相关性: 基于检索结果匹配度
            if retrieve_results:
                avg_retrieve_conf = (
                    sum(r.confidence for r in retrieve_results) / len(retrieve_results)
                )
                dim_scores["relevancy"] = avg_retrieve_conf
            else:
                dim_scores["relevancy"] = 0.8

            overall = sum(dim_scores.values()) / len(dim_scores) if dim_scores else 1.0
            result["score"] = overall
            result["dimensions"] = dim_scores

        except Exception as exc:  # noqa: BLE001
            logger.exception("质量评估异常")
            result["score"] = 0.5
            result["error"] = str(exc)

        return result

    def _run_conflict_detection(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """冲突检测 — 识别知识冲突."""
        result: dict[str, Any] = {
            "enabled": self._conflict_detector is not None,
            "score": 1.0,
            "conflicts_found": 0,
            "conflicts": [],
        }

        if self._conflict_detector is None:
            return result

        try:
            claims = self._extract_claims(execution_result)
            if not claims:
                return result

            conflicts: list[dict[str, Any]] = []

            retrieve_outputs = [
                r.output for r in execution_result.get_results_by_type(TaskType.RETRIEVE)
            ]
            reason_outputs = [
                r.output for r in execution_result.get_results_by_type(TaskType.REASON)
            ]

            if not any(self._has_results(o) for o in retrieve_outputs) and reason_outputs:
                conflicts.append({
                    "type": "retrieval_reason_mismatch",
                    "message": "检索无结果但推理产出了答案，可能存在幻觉风险",
                    "severity": "warning",
                })

            result["conflicts_found"] = len(conflicts)
            result["conflicts"] = conflicts
            result["score"] = max(0.0, 1.0 - len(conflicts) * 0.2)

        except Exception as exc:  # noqa: BLE001
            logger.exception("冲突检测异常")
            result["score"] = 0.5
            result["error"] = str(exc)

        return result

    def _run_compliance_check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """合规检查 — 策略与约束合规."""
        result: dict[str, Any] = {
            "enabled": True,
            "score": 1.0,
            "checks": [],
        }

        checks: list[dict[str, Any]] = []

        if execution_result.total_token_usage > 10000:
            checks.append({
                "check": "token_budget",
                "passed": False,
                "message": f"Token 使用 {execution_result.total_token_usage} 超出建议阈值",
                "severity": "warning",
            })

        if execution_result.total_elapsed_ms > 30000:
            checks.append({
                "check": "latency_budget",
                "passed": False,
                "message": f"延迟 {execution_result.total_elapsed_ms:.0f}ms 超出建议阈值",
                "severity": "warning",
            })

        if execution_result.fallback_triggered:
            checks.append({
                "check": "fallback_frequency",
                "passed": True,
                "message": "降级已触发，主计划执行失败",
                "severity": "info",
            })

        if len(execution_result.evidence_set) < 2:
            checks.append({
                "check": "evidence_sufficiency",
                "passed": False,
                "message": f"证据数量 {len(execution_result.evidence_set)} 不足",
                "severity": "warning",
            })

        result["checks"] = checks
        failed_checks = [c for c in checks if not c["passed"]]
        result["score"] = 1.0 - len(failed_checks) * 0.15

        return result

    # --------------------------------------------------------
    # 增强版聚合与辅助
    # --------------------------------------------------------

    def _aggregate_scores(
        self,
        fact_result: dict[str, Any],
        quality_result: dict[str, Any],
        conflict_result: dict[str, Any],
        compliance_result: dict[str, Any],
        faithfulness_result: dict[str, Any],
        consistency_result: dict[str, Any],
        strategy_result: dict[str, Any],
        domain_result: dict[str, Any] | None = None,
    ) -> float:
        """增强版聚合评分 (Discard-Weighted Voting).

        聚合八个维度:
        1. 事实校验 (fact_check)
        2. 质量评估 (quality)
        3. 冲突检测 (conflict)
        4. 合规检查 (compliance)
        5. Faithfulness (faithfulness)
        6. 自洽性 (self_consistency)
        7. 策略评估 (strategy_eval)
        8. 领域规则 (domain_rules)
        """
        scores = {
            "fact_check": fact_result.get("score", 1.0),
            "quality": quality_result.get("score", 1.0),
            "conflict": conflict_result.get("score", 1.0),
            "compliance": compliance_result.get("score", 1.0),
        }

        # 增强维度 (有条件加入)
        if faithfulness_result and "faithfulness_score" in faithfulness_result:
            scores["faithfulness"] = faithfulness_result["faithfulness_score"]
        else:
            scores["faithfulness"] = 1.0

        if consistency_result and "consistency_score" in consistency_result:
            scores["self_consistency"] = consistency_result["consistency_score"]
        else:
            scores["self_consistency"] = 1.0

        if strategy_result and "strategy_score" in strategy_result:
            scores["strategy_eval"] = strategy_result["strategy_score"]
        else:
            scores["strategy_eval"] = 1.0

        # 领域规则维度
        if domain_result and "overall_score" in domain_result:
            scores["domain_rules"] = domain_result["overall_score"]
        else:
            scores["domain_rules"] = 1.0

        total_weight = 0.0
        weighted_sum = 0.0

        for dim, score in scores.items():
            if score < self._discard_threshold:
                logger.warning(
                    "验证维度 %s 得分 %.2f 低于阈值 %.2f，丢弃",
                    dim, score, self._discard_threshold,
                )
                continue
            weight = self._weights.get(dim, 0.1)
            total_weight += weight
            weighted_sum += score * weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 4)

    @staticmethod
    def _score_to_status(score: float) -> ValidationSeverity:
        """将分数映射为验证状态."""
        if score >= 0.9:
            return ValidationSeverity.PASS
        if score >= 0.75:
            return ValidationSeverity.INFO
        if score >= 0.6:
            return ValidationSeverity.WARNING
        if score >= 0.4:
            return ValidationSeverity.ERROR
        return ValidationSeverity.CRITICAL

    def _collect_anomalies(
        self,
        fact_result: dict[str, Any],
        quality_result: dict[str, Any],
        conflict_result: dict[str, Any],
        compliance_result: dict[str, Any],
        faithfulness_result: dict[str, Any],
        consistency_result: dict[str, Any],
        domain_result: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """收集所有异常 (增强版)."""
        anomalies: list[dict[str, Any]] = []

        # 事实校验异常
        if not fact_result.get("passed", True):
            anomalies.append({
                "source": "fact_check",
                "message": f"事实校验未通过 (得分: {fact_result.get('score', 0):.2f})",
                "severity": "error",
            })

        # 冲突异常
        for conflict in conflict_result.get("conflicts", []):
            anomalies.append({
                "source": "conflict_detection",
                "message": conflict.get("message", ""),
                "severity": conflict.get("severity", "warning"),
            })

        # 合规异常
        for check in compliance_result.get("checks", []):
            if not check.get("passed", True):
                anomalies.append({
                    "source": "compliance",
                    "message": check.get("message", ""),
                    "severity": check.get("severity", "warning"),
                })

        # Faithfulness 异常
        if faithfulness_result and faithfulness_result.get("faithfulness_score", 1.0) < 0.6:
            anomalies.append({
                "source": "faithfulness",
                "message": (
                    f"Faithfulness 分数 {faithfulness_result.get('faithfulness_score', 0):.2f} 低于阈值, "
                    f"{len(faithfulness_result.get('unsupported_claims', []))} 条主张未被支持"
                ),
                "severity": "error" if faithfulness_result.get("faithfulness_score", 1.0) < 0.4 else "warning",
            })

        # 自洽性异常
        if consistency_result and consistency_result.get("contradictions"):
            anomalies.append({
                "source": "self_consistency",
                "message": (
                    f"发现 {len(consistency_result['contradictions'])} 处推理路径矛盾, "
                    f"自洽性分数 {consistency_result.get('consistency_score', 0):.2f}"
                ),
                "severity": "error",
            })

        # 领域规则异常
        if domain_result:
            for violation in domain_result.get("high_severity_violations", []):
                anomalies.append({
                    "source": "domain_rules",
                    "message": violation.get("issue", violation.get("type", "领域规则违规")),
                    "severity": violation.get("severity", "error"),
                    "details": violation,
                })

        return anomalies

    def _generate_recommendations(
        self,
        fact_result: dict[str, Any],
        quality_result: dict[str, Any],
        conflict_result: dict[str, Any],
        compliance_result: dict[str, Any],
        faithfulness_result: dict[str, Any],
        consistency_result: dict[str, Any],
        strategy_result: dict[str, Any],
        domain_result: dict[str, Any] | None = None,
    ) -> list[str]:
        """生成改进建议 (增强版)."""
        recommendations: list[str] = []

        # 事实校验建议
        failed_details = [
            d for d in fact_result.get("details", [])
            if d.get("status") == "failed"
        ]
        if failed_details:
            recommendations.append(
                f"事实校验: {len(failed_details)} 条声明未通过，建议核查标准值来源"
            )

        # 质量维度建议
        dimensions = quality_result.get("dimensions", {})
        weak_dims = [k for k, v in dimensions.items() if v < 0.6]
        if weak_dims:
            recommendations.append(
                f"质量评估: {', '.join(weak_dims)} 维度得分较低，建议针对性优化"
            )

        # 冲突建议
        if conflict_result.get("conflicts_found", 0) > 0:
            recommendations.append(
                "冲突检测: 发现知识冲突，建议启动冲突消解流程"
            )

        # 合规建议
        failed_checks = [
            c for c in compliance_result.get("checks", [])
            if not c.get("passed", True)
        ]
        if failed_checks:
            recommendations.append(
                f"合规检查: {len(failed_checks)} 项未通过，建议优化资源使用"
            )

        # Faithfulness 建议
        if faithfulness_result and faithfulness_result.get("unsupported_claims"):
            missing = faithfulness_result.get("missing_evidence", [])
            recommendations.append(
                f"Faithfulness: {len(faithfulness_result['unsupported_claims'])} 条主张未被检索上下文支持, "
                f"建议补充检索 ({len(missing)} 条补充查询建议)"
            )

        # 自洽性建议
        if consistency_result and consistency_result.get("contradictions"):
            recommendations.append(
                "自洽性: 推理路径存在矛盾，建议增加推理路径或采用加权投票"
            )

        # 策略评估建议
        if strategy_result and strategy_result.get("optimization_suggestions"):
            high_priority = [
                s for s in strategy_result["optimization_suggestions"]
                if s.get("priority") == "high"
            ]
            if high_priority:
                recommendations.append(
                    f"策略评估: {len(high_priority)} 项高优先级优化建议 "
                    f"({', '.join(s['dimension'] for s in high_priority[:3])})"
                )

        # 领域规则建议
        if domain_result:
            failed_rules = domain_result.get("failed_rules", 0)
            total_violations = len(domain_result.get("all_violations", []))
            if failed_rules > 0 or total_violations > 0:
                recommendations.append(
                    f"领域规则: {failed_rules} 条规则未通过, "
                    f"共 {total_violations} 项违规, "
                    f"建议核查领域知识一致性"
                )
            high_severity = domain_result.get("high_severity_violations", [])
            if high_severity:
                rule_ids = set()
                for v in high_severity:
                    if "rule_id" in v:
                        rule_ids.add(v["rule_id"])
                    if "type" in v:
                        rule_ids.add(v["type"])
                recommendations.append(
                    f"领域规则: {len(high_severity)} 项高严重级别违规 "
                    f"({', '.join(list(rule_ids)[:3])})"
                )

        return recommendations

    # --------------------------------------------------------
    # 内容提取辅助 (保持原有实现)
    # --------------------------------------------------------

    @staticmethod
    def _extract_verifiable_content(execution_result: ExecutionResult) -> str:
        """从执行结果中提取可校验的文本内容."""
        parts: list[str] = []

        for tr in execution_result.get_results_by_type(TaskType.REASON):
            answers = tr.output.get("answers", [])
            for ans in answers:
                if isinstance(ans, dict):
                    text = ans.get("text") or ans.get("value") or str(ans)
                    parts.append(text)
                elif isinstance(ans, str):
                    parts.append(ans)

        for tr in execution_result.get_results_by_type(TaskType.SYNTHESIZE):
            summary = tr.output.get("summary", "")
            if summary:
                parts.append(summary)

        return "\n".join(parts)

    @staticmethod
    def _extract_claims(execution_result: ExecutionResult) -> list[dict[str, Any]]:
        """从执行结果中提取声明列表."""
        claims: list[dict[str, Any]] = []

        for tr in execution_result.task_results.values():
            output = tr.output
            if "answers" in output:
                for ans in output["answers"]:
                    if isinstance(ans, dict):
                        claims.append({
                            "field": ans.get("param_name", "unknown"),
                            "value": ans.get("value", ans),
                            "source": tr.task_id,
                        })

        return claims

    @staticmethod
    def _has_results(output: dict[str, Any]) -> bool:
        """检查输出是否包含有效结果."""
        if not output:
            return False
        if "results" in output and output["results"]:
            return True
        if "answers" in output and output["answers"]:
            return True
        if "entities" in output and output["entities"]:
            return True
        return False


__all__ = [
    "ValidationOrchestrator",
]
