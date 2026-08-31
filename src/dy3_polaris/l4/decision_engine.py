"""L4 决策引擎层 — 顶层编排器 (DecisionEngine).

融合世界先进方案的端到端决策流程设计:
- LangGraph StateGraph: 状态化节点 + 条件边，完整闭环
- TDP 框架 (2026): Supervisor-Planner-Executor 三层上下文隔离
- PRISM MHCV: 多维度异构协同验证
- OLIVIA (2026): 上下文线性赌博机 + UCB 行动选择
- Plan-and-Solve: 计划 → 执行 → 验证 → 行动 → 反馈
- OutputSynthesizer: Platt Scaling 校准 + 安全感知输出

增强功能:
- 重试循环: 验证失败时按 max_iterations 重试
- V&R 闭环: 验证-精炼循环集成
- 输出合成: ActionRecord → OutputRecord 全链路
- 反馈驱动: Bayesian 估计 + 趋势检测

核心职责:
    串联 T1~T6 完整决策流程，提供单一入口处理用户查询。

流程:
    T1(IntentRouter) → T2(DecisionPlanner) → T3(TaskExecutor)
    → T4(ValidationOrchestrator) → T5(ActionSelector)
    → T5+(OutputSynthesizer) → T6(FeedbackAggregator)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from .action_selector import ActionSelector
from .adaptive_orchestrator import AdaptiveLearningOrchestrator
from .decision_planner import DecisionPlanner
from .feedback_aggregator import FeedbackAggregator
from .models import (
    ActionRecord,
    ActionType,
    DecisionPlan,
    ExecutionResult,
    ExecutionStatus,
    FeedbackSignal,
    FeedbackType,
    OutputRecord,
    ValidationReport,
    ValidationSeverity,
)
from .output_synthesizer import OutputSynthesizer
from .task_executor import TaskExecutor
from .validation_orchestrator import ValidationOrchestrator

logger = logging.getLogger(__name__)


class DecisionEngineConfig:
    """决策引擎配置."""

    def __init__(
        self,
        *,
        enable_validation: bool = True,
        enable_feedback: bool = True,
        enable_ucb: bool = True,
        rule_threshold: float = 0.5,
        ucb_exploration: float = 1.414,
        max_iterations: int = 3,
        fallback_on_failure: bool = True,
        enable_output_synthesis: bool = True,
        enable_adaptive_learning: bool = True,
        strategy: str = "ucb",
    ) -> None:
        """初始化决策引擎配置.

        Args:
            enable_validation: 是否启用验证
            enable_feedback: 是否启用反馈聚合
            enable_ucb: 是否启用学习策略
            rule_threshold: 强制规则选择的验证分数阈值
            ucb_exploration: UCB 探索系数
            max_iterations: 最大重试迭代次数
            fallback_on_failure: 失败时是否触发降级
            enable_output_synthesis: 是否启用输出合成
            enable_adaptive_learning: 是否启用自适应学习 (漂移检测+冷启动+A/B测试)
            strategy: 行动选择策略 ("ucb" / "thompson" / "linucb" / "ensemble")
        """
        self.enable_validation = enable_validation
        self.enable_feedback = enable_feedback
        self.enable_ucb = enable_ucb
        self.rule_threshold = rule_threshold
        self.ucb_exploration = ucb_exploration
        self.max_iterations = max_iterations
        self.fallback_on_failure = fallback_on_failure
        self.enable_output_synthesis = enable_output_synthesis
        self.enable_adaptive_learning = enable_adaptive_learning
        self.strategy = strategy


class DecisionEngine:
    """决策引擎顶层编排器 — 串联 T1~T6+ 完整流程.

    增强版支持:
    - 多策略行动选择 (UCB / Thompson / LinUCB / Ensemble)
    - 输出合成 (Platt Scaling + 安全约束)
    - 重试循环 (max_iterations)
    - V&R 闭环 (验证-精炼)
    - Bayesian 反馈聚合

    Usage::

        engine = DecisionEngine(
            intent_router=intent_router,
            task_executor=task_executor,
            config=DecisionEngineConfig(
                strategy="ensemble",
                max_iterations=3,
                enable_output_synthesis=True,
            ),
        )
        action_record = await engine.process_query("Dy3+ 的激发态波长是多少？")
        output = engine.synthesize_output(action_record, execution_result, validation_report)
    """

    def __init__(
        self,
        intent_router: Any,
        task_executor: TaskExecutor,
        fact_checker: Any | None = None,
        quality_manager: Any | None = None,
        conflict_detector: Any | None = None,
        *,
        config: DecisionEngineConfig | None = None,
    ) -> None:
        """初始化决策引擎.

        Args:
            intent_router: T1 意图路由器 (IntentRouter)
            task_executor: T3 任务执行器 (TaskExecutor)
            fact_checker: 事实校验器 (可选)
            quality_manager: 质量管理器 (可选)
            conflict_detector: 冲突检测器 (可选)
            config: 引擎配置
        """
        self._intent_router = intent_router
        self._task_executor = task_executor
        self._config = config or DecisionEngineConfig()

        # T2: 决策计划生成器
        self._decision_planner = DecisionPlanner()

        # T4: 验证编排器
        self._validation_orchestrator: ValidationOrchestrator | None = None
        if self._config.enable_validation:
            self._validation_orchestrator = ValidationOrchestrator(
                fact_checker=fact_checker,
                quality_manager=quality_manager,
                conflict_detector=conflict_detector,
            )

        # T5: 行动选择器 (多策略)
        self._action_selector = ActionSelector(
            use_ucb=self._config.enable_ucb,
            ucb_exploration=self._config.ucb_exploration,
            rule_threshold=self._config.rule_threshold,
            strategy=self._config.strategy,
        )

        # T5+: 输出合成器
        self._output_synthesizer: OutputSynthesizer | None = None
        if self._config.enable_output_synthesis:
            self._output_synthesizer = OutputSynthesizer()

        # T6: 反馈聚合器
        self._feedback_aggregator: FeedbackAggregator | None = None
        if self._config.enable_feedback:
            self._feedback_aggregator = FeedbackAggregator()

        # T6+: 自适应学习编排器
        self._adaptive_orchestrator: AdaptiveLearningOrchestrator | None = None
        if self._config.enable_adaptive_learning and self._config.enable_feedback:
            self._adaptive_orchestrator = AdaptiveLearningOrchestrator(
                feedback_aggregator=self._feedback_aggregator,
                action_selector=self._action_selector,
                output_synthesizer=self._output_synthesizer,
                auto_strategy_switch=True,
            )

        # 学习策略决策缓存 (next-action, 供反馈闭环引用)
        self._last_next_action: dict[str, dict[str, Any]] = {}

        logger.info(
            "DecisionEngine 初始化完成 (validation=%s, feedback=%s, ucb=%s, 策略=%s, 输出合成=%s, 自适应=%s)",
            self._config.enable_validation,
            self._config.enable_feedback,
            self._config.enable_ucb,
            self._config.strategy,
            self._config.enable_output_synthesis,
            self._adaptive_orchestrator is not None,
        )

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    async def process_query(
        self,
        query: str,
        *,
        context_id: str = "",
        learner_profile: dict[str, Any] | None = None,
        query_vector: list[float] | None = None,
    ) -> ActionRecord:
        """处理完整查询流程 (T1~T5)，支持重试循环.

        Args:
            query: 用户查询文本
            context_id: 上下文 ID
            learner_profile: 学习者画像
            query_vector: 预计算的查询向量

        Returns:
            ActionRecord 最终行动记录
        """
        start_ts = time.perf_counter()

        # T1: 意图理解与上下文构建
        routed_result = self._intent_router.route(query)
        intent_type = self._extract_intent_type(routed_result)
        logger.info("T1 完成: intent=%s", intent_type)

        # T2: 规划与任务分解
        plan = self._decision_planner.plan(
            routed_result,
            context_id=context_id,
            learner_profile=learner_profile,
        )
        logger.info("T2 完成: plan_id=%s, 子任务=%d", plan.plan_id, len(plan.sub_tasks))

        # T3+T4: 执行 + 验证 (带重试循环)
        execution_result, validation_report = await self._execute_with_retry(
            plan, query_vector=query_vector
        )

        # T5: 行动选择
        action_record = self._action_selector.select(validation_report, execution_result)
        action_record.plan_id = plan.plan_id
        logger.info(
            "T5 完成: action=%s, 置信度=%.4f",
            action_record.action_type.value,
            action_record.confidence,
        )

        # 记录总耗时
        total_elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        action_record.response_payload["_meta"] = {
            "plan_id": plan.plan_id,
            "intent_type": intent_type,
            "total_elapsed_ms": total_elapsed_ms,
            "execution_status": execution_result.status.value,
            "validation_status": validation_report.overall_status.value,
            "validation_score": validation_report.overall_score,
            "retry_count": validation_report.refinement_iterations,
            "strategy": self._config.strategy,
        }

        logger.info(
            "决策引擎处理完成: query=%s..., 总耗时=%.2fms, 行动=%s, 重试=%d",
            query[:40], total_elapsed_ms, action_record.action_type.value,
            validation_report.refinement_iterations,
        )

        return action_record

    # --------------------------------------------------------
    # 输出合成
    # --------------------------------------------------------

    def synthesize_output(
        self,
        action_record: ActionRecord,
        execution_result: ExecutionResult,
        validation_report: ValidationReport,
        *,
        intent_type: str = "",
    ) -> OutputRecord | None:
        """合成最终输出 (T5+).

        Args:
            action_record: 行动记录
            execution_result: 执行结果
            validation_report: 验证报告
            intent_type: 意图类型

        Returns:
            OutputRecord 或 None (输出合成未启用)
        """
        if self._output_synthesizer is None:
            logger.debug("输出合成器未启用")
            return None

        output = self._output_synthesizer.synthesize(
            action_record=action_record,
            execution_result=execution_result,
            validation_report=validation_report,
            intent_type=intent_type,
        )
        logger.info(
            "T5+ 输出合成: format=%s, 校准置信度=%.4f, 安全=%s",
            output.output_format.value,
            output.calibrated_confidence,
            output.safety_level.value,
        )
        return output

    async def process_query_full(
        self,
        query: str,
        *,
        context_id: str = "",
        learner_profile: dict[str, Any] | None = None,
        query_vector: list[float] | None = None,
    ) -> tuple[ActionRecord, OutputRecord | None]:
        """处理完整查询流程 (T1~T5+)，包含输出合成.

        Args:
            query: 用户查询文本
            context_id: 上下文 ID
            learner_profile: 学习者画像
            query_vector: 预计算的查询向量

        Returns:
            (ActionRecord, OutputRecord | None)
        """
        action_record = await self.process_query(
            query,
            context_id=context_id,
            learner_profile=learner_profile,
            query_vector=query_vector,
        )

        # 提取 meta 中的信息
        meta = action_record.response_payload.get("_meta", {})
        intent_type = meta.get("intent_type", "")

        # 重建 execution_result 和 validation_report 用于输出合成
        # (在实际应用中，这些应该缓存在引擎中)
        # 这里从 action_record 的 payload 中提取可用信息
        execution_result = self._reconstruct_execution_result(action_record)
        validation_report = self._reconstruct_validation_report(action_record)

        output_record = self.synthesize_output(
            action_record,
            execution_result,
            validation_report,
            intent_type=intent_type,
        )

        return action_record, output_record

    # --------------------------------------------------------
    # 学习策略决策 (唯一策略决策点: next-action)
    # --------------------------------------------------------

    async def process_next_action(
        self,
        learner_id: str,
        *,
        mode: str = "default",
        learner_profile: dict[str, Any] | None = None,
        context_id: str = "",
    ) -> dict[str, Any]:
        """生成下一次学习行动 — 统一策略决策 (review/guide/assess/default 模式).

        统一决策语义: action_type + confidence + recommended_path (KP 步骤),
        不再分散于 L5 run_guidance / skill_executor.

        Args:
            learner_id: 学习者 ID.
            mode: 策略模式 (default/review/guide/assess).
            learner_profile: 学习者画像 (dict 形式, 来自 L2).
            context_id: 上下文 ID (L1 会话 ID).

        Returns:
            统一决策体:
            {action_type, confidence, recommended_path[], plan_id, mode, summary}
        """
        from dy3_polaris.l4.learning_strategy import generate_next_action

        profile = learner_profile or {}
        decision = generate_next_action(profile, mode=mode)
        plan_id = f"na-{uuid.uuid4().hex[:12]}"
        decision["plan_id"] = plan_id
        decision["learner_id"] = learner_id
        decision["context_id"] = context_id
        # 反馈闭环入口: 策略执行可被记录
        self._last_next_action[learner_id] = {
            **decision,
            "ts": time.time(),
        }
        return decision

    def next_action_sync(
        self,
        learner_id: str,
        *,
        mode: str = "default",
        learner_profile: dict[str, Any] | None = None,
        context_id: str = "",
    ) -> dict[str, Any]:
        """同步版 next-action (供 L5 进程内编排调用, 无 await).

        与 process_next_action 完全同语义, 便于 skill_executor 等同步环境委托.
        """
        from dy3_polaris.l4.learning_strategy import generate_next_action

        profile = learner_profile or {}
        decision = generate_next_action(profile, mode=mode)
        decision["plan_id"] = f"na-{uuid.uuid4().hex[:12]}"
        decision["learner_id"] = learner_id
        decision["context_id"] = context_id
        self._last_next_action[learner_id] = {
            **decision,
            "ts": time.time(),
        }
        return decision

    # --------------------------------------------------------
    # 反馈闭环
    # --------------------------------------------------------

    def record_feedback(
        self,
        action_record: ActionRecord,
        rating: float,
        *,
        comment: str = "",
        feedback_type: FeedbackType = FeedbackType.EXPLICIT_RATING,
        intent_type: str = "",
    ) -> FeedbackSignal | None:
        """记录用户反馈 (T6 入口).

        当自适应学习编排器启用时，反馈将通过编排器处理，
        自动触发漂移检测、冷启动管理和策略调整。

        Args:
            action_record: 行动记录
            rating: 评分 (-1 ~ 1)
            comment: 评论
            feedback_type: 反馈类型
            intent_type: 意图类型

        Returns:
            FeedbackSignal 或 None (反馈未启用)
        """
        if self._feedback_aggregator is None:
            logger.debug("反馈聚合器未启用，跳过反馈记录")
            return None

        # 自适应学习编排器路由
        if self._adaptive_orchestrator is not None:
            signal = self._adaptive_orchestrator.process_feedback(
                action_record,
                rating,
                comment=comment,
                feedback_type=feedback_type,
                intent_type=intent_type,
            )
            logger.info(
                "自适应反馈记录: plan_id=%s, rating=%.2f, action=%s",
                action_record.plan_id, rating, action_record.action_type.value,
            )
            return signal

        # 传统反馈路径 (无自适应学习)
        signal = FeedbackSignal(
            plan_id=action_record.plan_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            intent_type=intent_type,
            action_type=action_record.action_type.value,
            validation_score=action_record.validation_score,
            execution_confidence=action_record.execution_confidence,
        )

        self._feedback_aggregator.add_signal(signal)

        # 更新行动选择器
        self._action_selector.feedback(action_record.action_type, rating)

        logger.info(
            "反馈记录: plan_id=%s, rating=%.2f, action=%s",
            action_record.plan_id, rating, action_record.action_type.value,
        )

        return signal

    def get_feedback_summary(
        self, *, last_hours: float = 24.0
    ) -> dict[str, Any] | None:
        """获取反馈摘要."""
        if self._feedback_aggregator is None:
            return None

        summary = self._feedback_aggregator.summarize(last_hours=last_hours)
        if summary is None:
            return None

        return {
            "period_hours": last_hours,
            "total_signals": summary.total_signals,
            "avg_rating": summary.avg_rating,
            "by_action": summary.by_action,
            "by_intent": summary.by_intent,
            "adjustments": summary.adjustments,
        }

    def get_bayesian_estimates(self, *, last_hours: float = 168.0) -> dict[str, Any] | None:
        """获取 Bayesian 反馈估计."""
        if self._feedback_aggregator is None:
            return None
        return self._feedback_aggregator.get_bayesian_estimates(last_hours=last_hours)

    def detect_feedback_trend(self, *, window_size: int = 10) -> dict[str, Any] | None:
        """检测反馈趋势."""
        if self._feedback_aggregator is None:
            return None
        return self._feedback_aggregator.detect_trend(window_size=window_size)

    def update_calibrator(
        self,
        feedback_data: list[tuple[float, bool]],
    ) -> None:
        """从反馈数据更新 Platt Scaling 校准器.

        Args:
            feedback_data: [(raw_confidence, actual_correct), ...]
        """
        if self._output_synthesizer is not None:
            self._output_synthesizer.update_calibrator(feedback_data)
            logger.info("Platt Scaling 校准器已更新: %d 条数据", len(feedback_data))

    # --------------------------------------------------------
    # 自适应学习接口
    # --------------------------------------------------------

    def get_adaptive_status(self) -> dict[str, Any] | None:
        """获取自适应学习系统状态.

        Returns:
            包含漂移检测、冷启动、策略信息的字典，或 None (自适应学习未启用)
        """
        if self._adaptive_orchestrator is None:
            return None
        return self._adaptive_orchestrator.get_system_summary()

    def get_adaptive_recommendations(self) -> dict[str, Any] | None:
        """获取自适应推荐.

        Returns:
            推荐字典，或 None (自适应学习未启用)
        """
        if self._adaptive_orchestrator is None:
            return None
        return self._adaptive_orchestrator.get_adaptive_recommendations()

    def is_drift_detected(self) -> bool:
        """是否检测到概念漂移."""
        if self._adaptive_orchestrator is None:
            return False
        return self._adaptive_orchestrator.is_drift_detected()

    def is_in_cold_start(self) -> bool:
        """是否处于冷启动阶段."""
        if self._adaptive_orchestrator is None:
            return False
        return self._adaptive_orchestrator.is_in_cold_start()

    def start_strategy_experiment(
        self,
        name: str,
        variants: list[str],
        *,
        min_samples: int = 30,
        significance_level: float = 0.05,
    ) -> str | None:
        """启动策略 A/B 测试实验.

        Args:
            name: 实验名称
            variants: 变体名称列表
            min_samples: 每变体最小样本数
            significance_level: 显著性水平

        Returns:
            实验 ID，或 None (自适应学习未启用)
        """
        if self._adaptive_orchestrator is None:
            return None
        return self._adaptive_orchestrator.start_strategy_experiment(
            name=name,
            variants=variants,
            min_samples=min_samples,
            significance_level=significance_level,
        )

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    async def _execute_with_retry(
        self,
        plan: DecisionPlan,
        *,
        query_vector: list[float] | None = None,
    ) -> tuple[ExecutionResult, ValidationReport]:
        """执行 + 验证 (带重试循环).

        利用 config.max_iterations 控制最大重试次数，
        利用 config.fallback_on_failure 控制是否触发降级。
        """
        max_iter = self._config.max_iterations
        exec_context: dict[str, Any] = {}
        if query_vector is not None:
            exec_context["query_vector"] = query_vector

        execution_result = await self._task_executor.execute(plan, context=exec_context)
        validation_report = self._run_validation(execution_result)

        # 重试循环: 验证未通过且有剩余迭代次数
        iteration = 0
        while (
            not validation_report.is_valid
            and iteration < max_iter
            and self._config.fallback_on_failure
        ):
            iteration += 1
            logger.info(
                "重试 %d/%d: 验证状态=%s, 分数=%.4f",
                iteration, max_iter,
                validation_report.overall_status.value,
                validation_report.overall_score,
            )

            # 降级: 使用简化计划
            if plan.fallback_plan is not None:
                fallback = plan.fallback_plan
                if fallback.simplified_tasks:
                    plan = DecisionPlan(
                        plan_id=f"{plan.plan_id}-retry-{iteration}",
                        sub_tasks=fallback.simplified_tasks,
                        execution_mode=fallback.fallback_mode,
                        original_query=plan.original_query,
                        context_id=plan.context_id,
                    )
                    execution_result = await self._task_executor.execute(plan, context=exec_context)
                else:
                    # 直接检索降级
                    execution_result.fallback_triggered = True
                    break
            else:
                # 无降级计划，直接重试
                execution_result = await self._task_executor.execute(plan, context=exec_context)

            validation_report = self._run_validation(execution_result)
            validation_report.refinement_iterations = iteration

        if iteration > 0:
            logger.info(
                "重试完成: 迭代=%d, 最终状态=%s, 分数=%.4f",
                iteration,
                validation_report.overall_status.value,
                validation_report.overall_score,
            )

        return execution_result, validation_report

    def _run_validation(self, execution_result: ExecutionResult) -> ValidationReport:
        """执行验证 (T4)."""
        if self._validation_orchestrator is not None:
            return self._validation_orchestrator.validate(execution_result)

        # 验证禁用: 生成默认通过报告
        report = ValidationReport(plan_id=execution_result.plan_id)
        report.overall_score = execution_result.confidence
        if execution_result.status == ExecutionStatus.FAILED:
            report.overall_status = ValidationSeverity.ERROR
        return report

    @staticmethod
    def _reconstruct_execution_result(action_record: ActionRecord) -> ExecutionResult:
        """从 ActionRecord 重建 ExecutionResult (简化版)."""
        meta = action_record.response_payload.get("_meta", {})
        result = ExecutionResult(
            plan_id=action_record.plan_id,
            status=ExecutionStatus.COMPLETED,
            confidence=action_record.execution_confidence,
        )
        result.reasoning_chain = action_record.response_payload.get("reasoning_chain", [])
        result.evidence_set = action_record.response_payload.get("evidence", [])
        return result

    @staticmethod
    def _reconstruct_validation_report(action_record: ActionRecord) -> ValidationReport:
        """从 ActionRecord 重建 ValidationReport (简化版)."""
        report = ValidationReport(
            plan_id=action_record.plan_id,
            overall_score=action_record.validation_score,
        )
        return report

    @staticmethod
    def _extract_intent_type(routed_result: Any) -> str:
        """从路由结果中提取意图类型."""
        if hasattr(routed_result, "intent") and hasattr(routed_result.intent, "intent_type"):
            return str(routed_result.intent.intent_type.value)
        if isinstance(routed_result, dict):
            intent = routed_result.get("intent", {})
            if isinstance(intent, dict):
                return intent.get("intent_type", "unknown")
            return str(intent)
        return "unknown"


__all__ = [
    "DecisionEngine",
    "DecisionEngineConfig",
]
