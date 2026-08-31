"""L4 决策引擎层 — 策略评估模块.

借鉴世界先进方案:
- PRISM (2026): 增益分解理论 — 探索增益 + 信息增益 + 聚合增益
- HydraRAG (2025): 三因子评分 — 源可信度 + 跨源 corroboration + 实体-证据对齐
- OLIVIA (2026): 上下文线性赌博机 — 基于反馈的策略优化

核心职责:
    不仅验证结果正确性，还评估推理策略本身的优劣:
    1. 检索策略评估: 检索策略选择是否合理、检索质量是否达标
    2. 推理策略评估: 推理模式选择是否恰当、推理链是否充分
    3. 资源效率评估: Token 使用、延迟控制是否高效
    4. 策略优化建议: 基于评估结果生成策略调整建议

输出:
    StrategyEvaluation — 策略评估结果，用于反馈到 T2 优化后续计划
"""

from __future__ import annotations

import logging
from typing import Any

from .models import (
    DecisionPlan,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    ReasoningMode,
    RetrievalStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """策略评估器 — 评估推理策略优劣并生成优化建议.

    借鉴 PRISM 增益分解理论，从三个维度评估策略:
    - 探索增益: 是否充分探索了解空间 (多检索策略/多推理路径)
    - 信息增益: 是否获取了高质量信息 (检索精度/证据质量)
    - 聚合增益: 是否高效聚合了信息 (推理效率/资源利用)

    Usage::

        evaluator = StrategyEvaluator()
        evaluation = evaluator.evaluate(plan, execution_result)
        # evaluation.strategy_score  -> 策略综合评分
        # evaluation.optimization_suggestions -> 优化建议列表
    """

    def __init__(
        self,
        *,
        token_budget: int = 8000,
        latency_budget_ms: float = 5000.0,
    ) -> None:
        """初始化策略评估器.

        Args:
            token_budget: Token 预算参考值
            latency_budget_ms: 延迟预算参考值 (毫秒)
        """
        self._token_budget = token_budget
        self._latency_budget = latency_budget_ms

        logger.info(
            "StrategyEvaluator 初始化 (Token 预算: %d, 延迟预算: %.0fms)",
            token_budget, latency_budget_ms,
        )

    def evaluate(
        self,
        plan: DecisionPlan | None,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """评估推理策略.

        Args:
            plan: T2 生成的决策计划 (可能为 None)
            execution_result: T3 执行结果

        Returns:
            策略评估结果字典:
            - strategy_score: 策略综合评分 (0~1)
            - exploration_gain: 探索增益 (0~1)
            - information_gain: 信息增益 (0~1)
            - aggregation_gain: 聚合增益 (0~1)
            - retrieval_strategy_assessment: 检索策略评估
            - reasoning_strategy_assessment: 推理策略评估
            - resource_efficiency: 资源效率评估
            - optimization_suggestions: 优化建议列表
        """
        # 评估三个增益维度
        exploration = self._assess_exploration_gain(plan, execution_result)
        information = self._assess_information_gain(execution_result)
        aggregation = self._assess_aggregation_gain(plan, execution_result)

        # 策略综合评分
        strategy_score = 0.3 * exploration + 0.4 * information + 0.3 * aggregation

        # 子维度评估
        retrieval_assessment = self._assess_retrieval_strategy(plan, execution_result)
        reasoning_assessment = self._assess_reasoning_strategy(plan, execution_result)
        resource_efficiency = self._assess_resource_efficiency(execution_result)

        # 生成优化建议
        suggestions = self._generate_suggestions(
            exploration, information, aggregation,
            retrieval_assessment, reasoning_assessment, resource_efficiency,
        )

        return {
            "strategy_score": round(strategy_score, 4),
            "exploration_gain": round(exploration, 4),
            "information_gain": round(information, 4),
            "aggregation_gain": round(aggregation, 4),
            "retrieval_strategy_assessment": retrieval_assessment,
            "reasoning_strategy_assessment": reasoning_assessment,
            "resource_efficiency": resource_efficiency,
            "optimization_suggestions": suggestions,
        }

    # --------------------------------------------------------
    # 增益评估
    # --------------------------------------------------------

    @staticmethod
    def _assess_exploration_gain(
        plan: DecisionPlan | None,
        result: ExecutionResult,
    ) -> float:
        """评估探索增益 — 是否充分探索了解空间.

        借鉴 PRISM 探索增益:
        - 多检索策略: 是否使用了多种检索策略
        - 多推理路径: 是否生成了多条推理路径
        - 实体覆盖: 是否检索到了足够的实体
        """
        score = 0.0

        # 检索策略多样性
        if plan:
            retrieve_tasks = [
                t for t in plan.sub_tasks if t.task_type == TaskType.RETRIEVE
            ]
            strategies_used = set()
            for t in retrieve_tasks:
                strategies_used.add(t.retrieval_strategy.value)
            strategy_diversity = min(1.0, len(strategies_used) / 2.0)
            score += 0.3 * strategy_diversity

            # 推理路径数量
            reason_tasks = [
                t for t in plan.sub_tasks if t.task_type == TaskType.REASON
            ]
            path_score = min(1.0, len(reason_tasks) / 2.0)
            score += 0.3 * path_score

            # 并行执行模式加分
            if plan.execution_mode == ExecutionMode.PARALLEL:
                score += 0.2
        else:
            score += 0.4  # 无计划信息时给中等分

        # 实体覆盖
        retrieve_results = result.get_results_by_type(TaskType.RETRIEVE)
        total_entities = 0
        for tr in retrieve_results:
            results = tr.output.get("results", [])
            total_entities += len(results)
        entity_coverage = min(1.0, total_entities / 5.0)
        score += 0.2 * entity_coverage

        return min(1.0, score)

    @staticmethod
    def _assess_information_gain(result: ExecutionResult) -> float:
        """评估信息增益 — 是否获取了高质量信息.

        借鉴 PRISM 信息增益:
        - 检索精度: 检索结果与查询的相关性
        - 证据质量: 证据的数量和可信度
        - 推理置信度: 推理结果的可信度
        """
        score = 0.0

        # 检索精度
        retrieve_results = result.get_results_by_type(TaskType.RETRIEVE)
        if retrieve_results:
            avg_conf = sum(r.confidence for r in retrieve_results) / len(retrieve_results)
            score += 0.3 * avg_conf
        else:
            score += 0.1

        # 证据质量
        evidence_count = len(result.evidence_set)
        evidence_score = min(1.0, evidence_count / 5.0)
        score += 0.3 * evidence_score

        # 推理置信度
        reason_results = result.get_results_by_type(TaskType.REASON)
        if reason_results:
            avg_reason_conf = sum(r.confidence for r in reason_results) / len(reason_results)
            score += 0.4 * avg_reason_conf
        else:
            score += 0.2

        return min(1.0, score)

    def _assess_aggregation_gain(
        self,
        plan: DecisionPlan | None,
        result: ExecutionResult,
    ) -> float:
        """评估聚合增益 — 是否高效聚合了信息.

        借鉴 PRISM 聚合增益:
        - 推理效率: 推理链长度与质量的平衡
        - 资源利用: Token 和延迟的使用效率
        - 结果完整性: 是否产出了完整的合成结果
        """
        score = 0.0

        # 推理效率
        reason_results = result.get_results_by_type(TaskType.REASON)
        if reason_results:
            # 推理链长度适中为好
            avg_chain_len = sum(
                len(r.output.get("reasoning_chain", []))
                for r in reason_results
            ) / len(reason_results)
            if 2 <= avg_chain_len <= 5:
                score += 0.3
            elif avg_chain_len > 0:
                score += 0.2
            else:
                score += 0.1

        # 资源利用
        token_efficiency = 1.0 - min(1.0, result.total_token_usage / self._token_budget)
        latency_efficiency = 1.0 - min(1.0, result.total_elapsed_ms / self._latency_budget)
        score += 0.3 * (0.5 * token_efficiency + 0.5 * latency_efficiency)

        # 结果完整性
        synthesize_results = result.get_results_by_type(TaskType.SYNTHESIZE)
        if synthesize_results and result.status == ExecutionStatus.COMPLETED:
            score += 0.4
        elif result.status == ExecutionStatus.COMPLETED:
            score += 0.2
        else:
            score += 0.0

        return min(1.0, score)

    # --------------------------------------------------------
    # 子维度评估
    # --------------------------------------------------------

    @staticmethod
    def _assess_retrieval_strategy(
        plan: DecisionPlan | None,
        result: ExecutionResult,
    ) -> dict[str, Any]:
        """评估检索策略."""
        assessment: dict[str, Any] = {
            "strategies_used": [],
            "total_results": 0,
            "avg_confidence": 0.0,
            "quality_score": 0.0,
        }

        if plan:
            retrieve_tasks = [
                t for t in plan.sub_tasks if t.task_type == TaskType.RETRIEVE
            ]
            assessment["strategies_used"] = list({
                t.retrieval_strategy.value for t in retrieve_tasks
            })

        retrieve_results = result.get_results_by_type(TaskType.RETRIEVE)
        total_results = 0
        confidences = []
        for tr in retrieve_results:
            results = tr.output.get("results", [])
            total_results += len(results)
            confidences.append(tr.confidence)

        assessment["total_results"] = total_results
        assessment["avg_confidence"] = (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        )

        # 质量评分: 结果数量和置信度的综合
        count_score = min(1.0, total_results / 5.0)
        conf_score = assessment["avg_confidence"]
        assessment["quality_score"] = round(0.5 * count_score + 0.5 * conf_score, 4)

        return assessment

    @staticmethod
    def _assess_reasoning_strategy(
        plan: DecisionPlan | None,
        result: ExecutionResult,
    ) -> dict[str, Any]:
        """评估推理策略."""
        assessment: dict[str, Any] = {
            "modes_used": [],
            "total_paths": 0,
            "avg_confidence": 0.0,
            "avg_chain_length": 0.0,
            "quality_score": 0.0,
        }

        if plan:
            reason_tasks = [
                t for t in plan.sub_tasks if t.task_type == TaskType.REASON
            ]
            assessment["modes_used"] = list({
                t.reasoning_mode.value for t in reason_tasks
            })

        reason_results = result.get_results_by_type(TaskType.REASON)
        assessment["total_paths"] = len(reason_results)

        if reason_results:
            confidences = [r.confidence for r in reason_results]
            chain_lengths = [
                len(r.output.get("reasoning_chain", []))
                for r in reason_results
            ]
            assessment["avg_confidence"] = round(
                sum(confidences) / len(confidences), 4
            )
            assessment["avg_chain_length"] = round(
                sum(chain_lengths) / len(chain_lengths), 2
            )

            # 质量评分
            conf_score = assessment["avg_confidence"]
            chain_score = min(1.0, assessment["avg_chain_length"] / 3.0)
            assessment["quality_score"] = round(
                0.6 * conf_score + 0.4 * chain_score, 4
            )

        return assessment

    def _assess_resource_efficiency(
        self, result: ExecutionResult
    ) -> dict[str, Any]:
        """评估资源效率."""
        token_usage = result.total_token_usage
        latency_ms = result.total_elapsed_ms

        token_efficiency = max(0.0, 1.0 - token_usage / self._token_budget)
        latency_efficiency = max(0.0, 1.0 - latency_ms / self._latency_budget)

        # 单位证据的 token 消耗
        evidence_count = max(1, len(result.evidence_set))
        token_per_evidence = token_usage / evidence_count

        return {
            "token_usage": token_usage,
            "token_budget": self._token_budget,
            "token_efficiency": round(token_efficiency, 4),
            "latency_ms": round(latency_ms, 2),
            "latency_budget_ms": self._latency_budget,
            "latency_efficiency": round(latency_efficiency, 4),
            "token_per_evidence": round(token_per_evidence, 2),
            "overall_efficiency": round(
                0.5 * token_efficiency + 0.5 * latency_efficiency, 4
            ),
        }

    # --------------------------------------------------------
    # 优化建议生成
    # --------------------------------------------------------

    @staticmethod
    def _generate_suggestions(
        exploration: float,
        information: float,
        aggregation: float,
        retrieval_assessment: dict[str, Any],
        reasoning_assessment: dict[str, Any],
        resource_efficiency: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """生成策略优化建议."""
        suggestions: list[dict[str, Any]] = []

        # 探索增益建议
        if exploration < 0.5:
            suggestions.append({
                "dimension": "exploration",
                "priority": "high" if exploration < 0.3 else "medium",
                "suggestion": "增加检索策略多样性，考虑并行使用 Hybrid + GraphRAG 检索",
                "current_score": round(exploration, 4),
                "target_score": 0.7,
            })

        # 信息增益建议
        if information < 0.5:
            suggestions.append({
                "dimension": "information",
                "priority": "high" if information < 0.3 else "medium",
                "suggestion": "提升检索质量，考虑增加重排序步骤或优化检索查询",
                "current_score": round(information, 4),
                "target_score": 0.7,
            })

        # 聚合增益建议
        if aggregation < 0.5:
            suggestions.append({
                "dimension": "aggregation",
                "priority": "medium",
                "suggestion": "优化推理链长度和资源利用效率",
                "current_score": round(aggregation, 4),
                "target_score": 0.7,
            })

        # 检索策略建议
        if retrieval_assessment["quality_score"] < 0.5:
            suggestions.append({
                "dimension": "retrieval",
                "priority": "high",
                "suggestion": f"检索质量评分 {retrieval_assessment['quality_score']:.2f} 较低，建议优化 chunk 大小或增加检索结果数量",
                "current_score": retrieval_assessment["quality_score"],
                "target_score": 0.7,
            })

        # 推理策略建议
        if reasoning_assessment["quality_score"] < 0.5:
            suggestions.append({
                "dimension": "reasoning",
                "priority": "medium",
                "suggestion": f"推理质量评分 {reasoning_assessment['quality_score']:.2f} 较低，建议增加推理步骤或多路径验证",
                "current_score": reasoning_assessment["quality_score"],
                "target_score": 0.7,
            })

        # 资源效率建议
        if resource_efficiency["overall_efficiency"] < 0.5:
            suggestions.append({
                "dimension": "resource",
                "priority": "low",
                "suggestion": f"资源效率 {resource_efficiency['overall_efficiency']:.2f} 有优化空间，Token 使用 {resource_efficiency['token_usage']}, 延迟 {resource_efficiency['latency_ms']:.0f}ms",
                "current_score": resource_efficiency["overall_efficiency"],
                "target_score": 0.7,
            })

        return suggestions


__all__ = [
    "StrategyEvaluator",
]
