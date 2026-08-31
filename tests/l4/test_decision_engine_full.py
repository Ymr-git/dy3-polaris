"""L4 决策引擎完整单元测试套件.

测试范围:
- DecisionPlanner (T2): 计划生成、模板选择、资源估算
- ValidationOrchestrator (T4): 多维度验证、评分聚合、异常检测
- ActionSelector (T5): 规则选择、UCB选择、响应构建
- FeedbackAggregator (T6): 信号聚合、时间衰减、策略建议
- DecisionEngine (顶层): 端到端流程串联、配置控制、反馈闭环
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from dy3_polaris.l4 import (
    ActionRecord,
    ActionSelector,
    ActionType,
    DecisionEngine,
    DecisionEngineConfig,
    DecisionPlanner,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FeedbackAggregator,
    FeedbackSignal,
    FeedbackType,
    PlanTemplate,
    ResourceBudget,
    RuleBasedSelector,
    SubTask,
    TaskResult,
    TaskType,
    UCBActionSelector,
    ValidationOrchestrator,
    ValidationReport,
    ValidationSeverity,
)
from dy3_polaris.l4.models import DecisionPlan, ReasoningMode, RetrievalStrategy
from dy3_polaris.l4.task_executor import TaskExecutor


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_graph_reasoner() -> MagicMock:
    r = MagicMock()
    r.reason.return_value = MagicMock(
        mode=MagicMock(value="multi_hop"),
        answers=[{"entity_id": "Dy3+", "confidence": 0.9}],
        confidence=0.9,
        reasoning_chain=["多跳推理测试"],
        evidence_triples=[],
        elapsed_ms=15.0,
    )
    return r


@pytest.fixture
def mock_hybrid_retriever() -> MagicMock:
    r = MagicMock()
    r.retrieve.return_value = MagicMock(
        results=[{"chunk_id": "c1", "content": "测试内容"}],
        scores=[0.95],
        total=1,
        source_type="hybrid",
    )
    return r


@pytest.fixture
def mock_graphrag_retriever() -> MagicMock:
    r = MagicMock()
    r.search.return_value = MagicMock(
        local_result=MagicMock(model_dump=lambda: {"entities": ["Dy3+"]}),
        global_result=None,
        strategy="adaptive",
    )
    return r


@pytest.fixture
def mock_intent_router() -> MagicMock:
    r = MagicMock()
    r.route.return_value = MagicMock(
        intent=MagicMock(intent_type=MagicMock(value="concept")),
        query="Dy3+ 的发光原理是什么",
        retrieval_result=MagicMock(query="Dy3+ 的发光原理是什么"),
    )
    return r


@pytest.fixture
def executor(
    mock_store: MagicMock,
    mock_graph_reasoner: MagicMock,
    mock_hybrid_retriever: MagicMock,
    mock_graphrag_retriever: MagicMock,
) -> TaskExecutor:
    return TaskExecutor(
        store=mock_store,
        graph_reasoner=mock_graph_reasoner,
        hybrid_retriever=mock_hybrid_retriever,
        graphrag_retriever=mock_graphrag_retriever,
    )


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """构建一个示例 ExecutionResult."""
    result = ExecutionResult(plan_id="plan-test", status=ExecutionStatus.COMPLETED)
    result.task_results["retrieve"] = TaskResult(
        task_id="retrieve",
        task_type=TaskType.RETRIEVE,
        status=ExecutionStatus.COMPLETED,
        output={"results": [{"chunk_id": "c1"}], "total": 1},
        confidence=0.85,
        evidence=[{"type": "chunk"}],
        reasoning_chain=["检索: 1 条结果"],
    )
    result.task_results["reason"] = TaskResult(
        task_id="reason",
        task_type=TaskType.REASON,
        status=ExecutionStatus.COMPLETED,
        output={"answers": [{"text": "Dy3+ 激发态为 4F9/2"}], "confidence": 0.9},
        confidence=0.9,
        evidence=[{"type": "triple"}],
        reasoning_chain=["推理: 多跳推理完成"],
    )
    result.task_results["synthesize"] = TaskResult(
        task_id="synthesize",
        task_type=TaskType.SYNTHESIZE,
        status=ExecutionStatus.COMPLETED,
        output={"summary": "Dy3+ 的激发态..."},
        confidence=0.88,
        reasoning_chain=["合成: 响应生成"],
    )
    result.evidence_set = [{"type": "chunk"}, {"type": "triple"}]
    result.reasoning_chain = ["检索", "推理", "合成"]
    result.confidence = 0.88
    result.total_elapsed_ms = 150.0
    result.total_token_usage = 2048
    return result


@pytest.fixture
def failed_execution_result() -> ExecutionResult:
    """构建一个失败的 ExecutionResult."""
    result = ExecutionResult(plan_id="plan-fail", status=ExecutionStatus.FAILED)
    result.error_summary = "检索模块异常"
    return result


# ============================================================
# T2: DecisionPlanner 测试
# ============================================================


class TestDecisionPlanner:
    """决策计划生成器测试."""

    def test_plan_concept_intent(self) -> None:
        """测试概念查询计划生成."""
        planner = DecisionPlanner()
        routed = {
            "intent": {"intent_type": "concept", "extracted_entities": []},
            "query": "什么是能量传递",
        }
        plan = planner.plan(routed)

        assert isinstance(plan, DecisionPlan)
        assert plan.execution_mode == ExecutionMode.SEQUENTIAL
        assert len(plan.sub_tasks) == 2
        assert plan.sub_tasks[0].task_type == TaskType.RETRIEVE
        assert plan.sub_tasks[1].task_type == TaskType.SYNTHESIZE
        assert plan.sub_tasks[1].deps == ["retrieve_concept"]
        assert plan.original_query == "什么是能量传递"

    def test_plan_numeric_intent(self) -> None:
        """测试数值查询计划生成."""
        planner = DecisionPlanner()
        routed = {
            "intent": {"intent_type": "numeric", "extracted_entities": [{"entity_type": "ion", "text": "Dy3+"}]},
            "query": "Dy3+ 的激发态波长是多少",
        }
        plan = planner.plan(routed)

        assert plan.execution_mode == ExecutionMode.SEQUENTIAL
        assert len(plan.sub_tasks) == 4
        types = [t.task_type for t in plan.sub_tasks]
        assert types == [TaskType.RETRIEVE, TaskType.REASON, TaskType.VERIFY, TaskType.SYNTHESIZE]

    def test_plan_relational_intent(self) -> None:
        """测试关系查询计划生成."""
        planner = DecisionPlanner()
        routed = {
            "intent": {"intent_type": "relational", "extracted_entities": []},
            "query": "Dy3+ 和 Eu3+ 的关系",
        }
        plan = planner.plan(routed)

        assert len(plan.sub_tasks) == 3
        assert plan.sub_tasks[0].retrieval_strategy == RetrievalStrategy.SUBGRAPH
        assert plan.sub_tasks[1].reasoning_mode == ReasoningMode.PATH_FINDING

    def test_plan_composite_intent(self) -> None:
        """测试复合查询计划生成."""
        planner = DecisionPlanner()
        routed = {
            "intent": {"intent_type": "composite", "extracted_entities": []},
            "query": "比较 Dy3+ 和 Eu3+ 的发光效率并给出数值",
        }
        plan = planner.plan(routed)

        assert plan.execution_mode == ExecutionMode.PARALLEL
        assert len(plan.sub_tasks) == 6
        # 检查并行检索任务无互相依赖
        hybrid_task = next(t for t in plan.sub_tasks if t.task_id == "retrieve_hybrid")
        graphrag_task = next(t for t in plan.sub_tasks if t.task_id == "retrieve_graphrag")
        assert hybrid_task.deps == []
        assert graphrag_task.deps == []

    def test_plan_with_entity_id(self) -> None:
        """测试实体 ID 注入."""
        planner = DecisionPlanner()
        routed = {
            "intent": {"intent_type": "concept", "extracted_entities": [{"entity_type": "ion", "text": "Dy3+"}]},
            "query": "Dy3+ 发光",
        }
        plan = planner.plan(routed)
        assert plan.sub_tasks[0].params.get("entity_id") == "Dy3+"

    def test_plan_resource_estimation(self) -> None:
        """测试资源估算."""
        planner = DecisionPlanner()
        routed = {"intent": {"intent_type": "concept"}, "query": "测试"}
        plan = planner.plan(routed)

        assert plan.estimated_total_tokens > 0
        assert plan.estimated_total_latency_ms > 0
        assert plan.fallback_plan is not None

    def test_plan_unknown_intent_defaults_to_composite(self) -> None:
        """测试未知意图回退到复合查询."""
        planner = DecisionPlanner()
        routed = {"intent": {"intent_type": "unknown"}, "query": "测试"}
        plan = planner.plan(routed)
        assert len(plan.sub_tasks) == 6  # 复合查询模板

    def test_extract_from_routed_result_variations(self) -> None:
        """测试从不同结构提取信息."""
        planner = DecisionPlanner()

        # 字典结构
        routed_dict = {"intent": {"intent_type": "concept"}, "query": "Q1"}
        plan = planner.plan(routed_dict)
        assert plan.original_query == "Q1"

        # 对象结构
        routed_obj = MagicMock()
        routed_obj.intent = MagicMock()
        routed_obj.intent.intent_type = MagicMock(value="numeric")
        routed_obj.intent.extracted_entities = []
        routed_obj.retrieval_result = MagicMock(query="Q2")
        plan = planner.plan(routed_obj)
        assert plan.original_query == "Q2"


class TestPlanTemplate:
    """计划模板库测试."""

    def test_concept_query_template(self) -> None:
        tasks = PlanTemplate.concept_query("什么是能量传递")
        assert len(tasks) == 2
        assert tasks[0].task_id == "retrieve_concept"
        assert tasks[1].task_id == "synthesize_concept"
        assert tasks[1].deps == ["retrieve_concept"]

    def test_numeric_query_template(self) -> None:
        tasks = PlanTemplate.numeric_query("Dy3+ 波长", entity_id="Dy3+")
        assert len(tasks) == 4
        assert tasks[0].params.get("entity_id") == "Dy3+"
        assert tasks[2].task_type == TaskType.VERIFY

    def test_relational_query_template(self) -> None:
        tasks = PlanTemplate.relational_query("Dy3+ 和 Eu3+ 关系")
        assert tasks[0].retrieval_strategy == RetrievalStrategy.SUBGRAPH
        assert tasks[1].reasoning_mode == ReasoningMode.PATH_FINDING

    def test_composite_query_template(self) -> None:
        tasks = PlanTemplate.composite_query("比较 Dy3+ 和 Eu3+")
        assert len(tasks) == 6
        # 两个并行检索
        assert tasks[0].deps == []
        assert tasks[1].deps == []
        # 两个并行推理依赖两个检索
        assert set(tasks[2].deps) == {"retrieve_hybrid", "retrieve_graphrag"}
        assert set(tasks[3].deps) == {"retrieve_hybrid", "retrieve_graphrag"}


# ============================================================
# T4: ValidationOrchestrator 测试
# ============================================================


class TestValidationOrchestrator:
    """验证编排器测试."""

    def test_validate_success(self, sample_execution_result: ExecutionResult) -> None:
        """测试正常执行结果的验证."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(sample_execution_result)

        assert isinstance(report, ValidationReport)
        assert report.plan_id == "plan-test"
        assert report.overall_score > 0
        assert report.is_valid
        assert not report.needs_human_review
        assert report.validation_time_ms >= 0

    def test_validate_failed_execution(self, failed_execution_result: ExecutionResult) -> None:
        """测试失败执行结果的验证."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(failed_execution_result)

        assert report.overall_status == ValidationSeverity.ERROR
        assert report.overall_score == 0.0
        assert not report.is_valid
        assert len(report.anomalies) > 0

    def test_fact_check_with_mock(self, sample_execution_result: ExecutionResult) -> None:
        """测试事实校验集成."""
        mock_checker = MagicMock()
        mock_checker.check.return_value = MagicMock(
            confidence=0.95,
            overall_passed=True,
            checked=3,
            passed=3,
            failed=0,
            results=[{"text": "数值1", "status": "passed", "deviation": 0.0}],
        )

        orchestrator = ValidationOrchestrator(fact_checker=mock_checker)
        report = orchestrator.validate(sample_execution_result)

        assert report.fact_check["enabled"] is True
        assert report.fact_check["score"] == 0.95
        assert report.fact_check["passed"] is True
        mock_checker.check.assert_called_once()

    def test_quality_assessment(self, sample_execution_result: ExecutionResult) -> None:
        """测试质量评估."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(sample_execution_result)

        dims = report.quality_assessment.get("dimensions", {})
        assert "accuracy" in dims
        assert "consistency" in dims
        assert "completeness" in dims
        assert report.quality_assessment["score"] > 0

    def test_conflict_detection(self, sample_execution_result: ExecutionResult) -> None:
        """测试冲突检测."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(sample_execution_result)

        assert "conflicts_found" in report.conflict_detection
        assert report.conflict_detection["score"] >= 0

    def test_compliance_check(self, sample_execution_result: ExecutionResult) -> None:
        """测试合规检查."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(sample_execution_result)

        checks = report.compliance_check.get("checks", [])
        assert isinstance(checks, list)
        assert report.compliance_check["score"] >= 0

    def test_score_to_status_mapping(self) -> None:
        """测试分数到状态映射."""
        assert ValidationOrchestrator._score_to_status(0.95) == ValidationSeverity.PASS
        assert ValidationOrchestrator._score_to_status(0.8) == ValidationSeverity.INFO
        assert ValidationOrchestrator._score_to_status(0.65) == ValidationSeverity.WARNING
        assert ValidationOrchestrator._score_to_status(0.5) == ValidationSeverity.ERROR
        assert ValidationOrchestrator._score_to_status(0.3) == ValidationSeverity.CRITICAL

    def test_aggregate_scores_discard_low(self) -> None:
        """测试丢弃低分维度的聚合."""
        orchestrator = ValidationOrchestrator(discard_threshold=0.5)
        fact = {"score": 0.9}
        quality = {"score": 0.3}  # 低于阈值，应被丢弃
        conflict = {"score": 0.8}
        compliance = {"score": 0.85}

        score = orchestrator._aggregate_scores(
            fact, quality, conflict, compliance,
            {}, {}, {}, None,
        )
        # quality 被丢弃，剩余维度加权
        assert score > 0
        assert score <= 1.0

    def test_recommendations_generation(self, sample_execution_result: ExecutionResult) -> None:
        """测试改进建议生成."""
        orchestrator = ValidationOrchestrator()
        report = orchestrator.validate(sample_execution_result)

        assert isinstance(report.recommendations, list)
        # 正常通过时建议较少

    def test_anomalies_collection(self) -> None:
        """测试异常收集."""
        orchestrator = ValidationOrchestrator()
        anomalies = orchestrator._collect_anomalies(
            fact_result={"passed": False, "score": 0.3},
            quality_result={"dimensions": {"accuracy": 0.5}},
            conflict_result={"conflicts": [{"message": "冲突1", "severity": "error"}]},
            compliance_result={"checks": [{"passed": False, "message": "超时", "severity": "warning"}]},
            faithfulness_result={},
            consistency_result={},
            domain_result=None,
        )

        assert len(anomalies) >= 3  # fact + conflict + compliance


# ============================================================
# T5: ActionSelector 测试
# ============================================================


class TestUCBActionSelector:
    """UCB 行动选择器测试."""

    def test_select_explores_untried_first(self) -> None:
        """测试优先探索未尝试的行动."""
        selector = UCBActionSelector()
        action, score = selector.select({"validation_score": 0.8})
        assert score == float("inf")
        # 第一次选择后，该行动有记录了
        selector.update(action, 0.5)

    def test_ucb_formula(self) -> None:
        """测试 UCB 计算公式."""
        selector = UCBActionSelector(exploration_constant=1.414)
        # 初始化所有行动
        for a in ActionType:
            selector.update(a, 0.5)

        context = {"validation_score": 0.8}
        action, score = selector.select(context)
        assert action in list(ActionType)
        assert score != float("inf")

    def test_update_and_average(self) -> None:
        """测试增量平均更新."""
        selector = UCBActionSelector()
        selector.update(ActionType.DIRECT_ANSWER, 1.0)
        selector.update(ActionType.DIRECT_ANSWER, 0.0)

        stats = selector.get_stats()
        assert stats["direct_answer"]["count"] == 2
        assert stats["direct_answer"]["avg_reward"] == 0.5


class TestRuleBasedSelector:
    """规则行动选择器测试."""

    def test_pass_goes_direct(self) -> None:
        """测试 PASS 状态直接回答."""
        report = ValidationReport(overall_status=ValidationSeverity.PASS, overall_score=0.95)
        result = ExecutionResult(plan_id="p1")
        action, reason = RuleBasedSelector.select(report, result)
        assert action == ActionType.DIRECT_ANSWER
        assert "通过" in reason

    def test_critical_goes_human(self) -> None:
        """测试 CRITICAL 状态转人工."""
        report = ValidationReport(overall_status=ValidationSeverity.CRITICAL, overall_score=0.2)
        result = ExecutionResult(plan_id="p1")
        action, reason = RuleBasedSelector.select(report, result)
        assert action == ActionType.HUMAN_CONFIRM
        assert "严重" in reason or "人工" in reason

    def test_error_with_high_confidence_goes_tool(self) -> None:
        """测试 ERROR 但高执行置信度时工具增强."""
        report = ValidationReport(overall_status=ValidationSeverity.ERROR, overall_score=0.5)
        result = ExecutionResult(plan_id="p1", confidence=0.7)
        action, reason = RuleBasedSelector.select(report, result)
        assert action == ActionType.TOOL_ENHANCED

    def test_error_with_low_confidence_goes_human(self) -> None:
        """测试 ERROR 且低执行置信度时转人工."""
        report = ValidationReport(overall_status=ValidationSeverity.ERROR, overall_score=0.5)
        result = ExecutionResult(plan_id="p1", confidence=0.3)
        action, reason = RuleBasedSelector.select(report, result)
        assert action == ActionType.HUMAN_CONFIRM

    def test_warning_goes_negotiate(self) -> None:
        """测试 WARNING 状态协商确认."""
        report = ValidationReport(overall_status=ValidationSeverity.WARNING, overall_score=0.65)
        result = ExecutionResult(plan_id="p1")
        action, reason = RuleBasedSelector.select(report, result)
        assert action == ActionType.NEGOTIATE


class TestActionSelector:
    """行动选择器测试."""

    def test_select_low_score_uses_rule(self) -> None:
        """测试低分场景强制使用规则选择."""
        selector = ActionSelector(rule_threshold=0.6)
        report = ValidationReport(overall_score=0.4, overall_status=ValidationSeverity.ERROR)
        result = ExecutionResult(plan_id="p1")

        record = selector.select(report, result)
        assert record.action_type == ActionType.HUMAN_CONFIRM
        assert record.confidence < 0.6

    def test_select_high_score_uses_ucb(self) -> None:
        """测试高分场景使用 UCB 选择."""
        selector = ActionSelector(use_ucb=True, rule_threshold=0.3)
        report = ValidationReport(overall_score=0.9, overall_status=ValidationSeverity.PASS)
        result = ExecutionResult(plan_id="p1", confidence=0.85)

        record = selector.select(report, result)
        assert record.action_type in list(ActionType)
        assert record.selection_reason.startswith("UCB")

    def test_build_payload_direct(self) -> None:
        """测试 DIRECT_ANSWER 响应载荷."""
        selector = ActionSelector()
        report = ValidationReport(overall_score=0.95, overall_status=ValidationSeverity.PASS)
        result = ExecutionResult(plan_id="p1")
        result.task_results["reason"] = TaskResult(
            task_id="reason", task_type=TaskType.REASON,
            output={"answers": [{"text": "答案"}]},
        )

        record = selector.select(report, result)
        assert record.response_payload["response_type"] == "direct"
        assert "answers" in record.response_payload

    def test_build_payload_tool_enhanced(self) -> None:
        """测试 TOOL_ENHANCED 响应载荷."""
        selector = ActionSelector(use_ucb=False)
        report = ValidationReport(
            overall_score=0.5, overall_status=ValidationSeverity.ERROR,
            fact_check={"passed": False},
        )
        result = ExecutionResult(plan_id="p1", confidence=0.6)

        record = selector.select(report, result)
        assert record.action_type == ActionType.TOOL_ENHANCED
        assert len(record.tool_calls) > 0

    def test_build_payload_negotiate(self) -> None:
        """测试 NEGOTIATE 响应载荷."""
        selector = ActionSelector(use_ucb=False)
        report = ValidationReport(
            overall_score=0.65, overall_status=ValidationSeverity.WARNING,
            anomalies=[{"message": "证据不足"}],
        )
        result = ExecutionResult(plan_id="p1")

        record = selector.select(report, result)
        assert record.action_type == ActionType.NEGOTIATE
        assert len(record.clarification_questions) > 0

    def test_feedback_updates_ucb(self) -> None:
        """测试反馈更新 UCB."""
        selector = ActionSelector(use_ucb=True)
        selector.feedback(ActionType.DIRECT_ANSWER, 1.0)

        stats = selector.get_ucb_stats()
        assert stats["direct_answer"]["count"] == 1
        assert stats["direct_answer"]["avg_reward"] == 1.0

    def test_build_clarification_questions(self) -> None:
        """测试澄清问题构建."""
        report = ValidationReport(
            anomalies=[
                {"message": "事实校验未通过"},
                {"message": "发现冲突"},
            ]
        )
        questions = ActionSelector._build_clarification_questions(report)
        assert len(questions) > 0
        assert len(questions) <= 3


# ============================================================
# T6: FeedbackAggregator 测试
# ============================================================


class TestFeedbackAggregator:
    """反馈聚合器测试."""

    def test_add_signal(self) -> None:
        """测试添加反馈信号."""
        aggregator = FeedbackAggregator(max_history=100)
        signal = FeedbackSignal(rating=0.8, action_type="direct_answer")
        aggregator.add_signal(signal)

        assert len(aggregator._signals) == 1
        assert len(aggregator._by_action["direct_answer"]) == 1

    def test_max_history_truncation(self) -> None:
        """测试历史超限裁剪."""
        aggregator = FeedbackAggregator(max_history=3)
        for i in range(5):
            aggregator.add_signal(FeedbackSignal(rating=0.5, action_type="direct_answer"))

        assert len(aggregator._signals) == 3

    def test_add_explicit_feedback(self) -> None:
        """测试添加显式反馈."""
        aggregator = FeedbackAggregator()
        record = ActionRecord(plan_id="p1", action_type=ActionType.DIRECT_ANSWER)
        signal = aggregator.add_explicit_feedback(record, rating=1.0, comment="很好")

        assert signal.feedback_type == FeedbackType.EXPLICIT_RATING
        assert signal.rating == 1.0
        assert signal.comment == "很好"

    def test_add_implicit_signal(self) -> None:
        """测试添加隐式信号."""
        aggregator = FeedbackAggregator()
        record = ActionRecord(plan_id="p1", action_type=ActionType.DIRECT_ANSWER)
        signal = aggregator.add_implicit_signal(record, "dwell_time", 45.0)

        assert signal.feedback_type == FeedbackType.IMPLICIT_SIGNAL
        assert signal.rating > 0  # 停留 45s 应为正反馈

    def test_summarize_insufficient_signals(self) -> None:
        """测试信号不足时返回 None."""
        aggregator = FeedbackAggregator()
        aggregator.add_signal(FeedbackSignal(rating=0.5))

        summary = aggregator.summarize(last_hours=24, min_signals=5)
        assert summary is None

    def test_summarize_with_enough_signals(self) -> None:
        """测试正常汇总."""
        aggregator = FeedbackAggregator()
        for _ in range(10):
            aggregator.add_signal(FeedbackSignal(
                rating=0.8,
                action_type="direct_answer",
                intent_type="concept",
            ))

        summary = aggregator.summarize(last_hours=24, min_signals=5)
        assert summary is not None
        assert summary.avg_rating == 0.8
        assert summary.by_action["direct_answer"]["count"] == 10
        assert summary.by_intent["concept"]["count"] == 10

    def test_time_decay_weight(self) -> None:
        """测试时间衰减权重."""
        aggregator = FeedbackAggregator(decay_half_life_hours=1.0)
        now = time.time()

        # 2 小时前的信号
        weight_old = aggregator._time_weight(now - 7200, now)
        # 现在的信号
        weight_new = aggregator._time_weight(now, now)

        assert weight_new == 1.0
        assert weight_old < 1.0
        assert weight_old > 0  # 指数衰减不会到 0

    def test_implicit_to_rating(self) -> None:
        """测试隐式信号转评分."""
        assert FeedbackAggregator._implicit_to_rating("dwell_time", 45.0) > 0
        assert FeedbackAggregator._implicit_to_rating("dwell_time", 3.0) < 0
        assert FeedbackAggregator._implicit_to_rating("copy", 1.0) == 0.8
        assert FeedbackAggregator._implicit_to_rating("correct", 1.0) == -1.0
        assert FeedbackAggregator._implicit_to_rating("skip", 1.0) == -0.5

    def test_generate_adjustments(self) -> None:
        """测试策略调整建议."""
        from dy3_polaris.l4.models import FeedbackSummary

        summary = FeedbackSummary(
            avg_rating=-0.4,
            by_action={
                "direct_answer": {"avg_rating": 0.8, "count": 10},
                "human_confirm": {"avg_rating": -0.5, "count": 10},
            },
        )
        adjustments = FeedbackAggregator._generate_adjustments(summary)

        assert len(adjustments) >= 2  # 高分行动+低分行动+全局低分
        # 检查是否包含对 human_confirm 的降低频率建议
        human_adj = [a for a in adjustments if a.get("action") == "human_confirm"]
        assert len(human_adj) > 0
        assert human_adj[0]["adjustment"] == "reduce_frequency"

    def test_get_action_rewards(self) -> None:
        """测试获取行动回报."""
        aggregator = FeedbackAggregator()
        aggregator.add_signal(FeedbackSignal(rating=0.9, action_type="direct_answer"))
        aggregator.add_signal(FeedbackSignal(rating=0.3, action_type="negotiate"))

        rewards = aggregator.get_action_rewards(last_hours=24)
        assert "direct_answer" in rewards
        assert "negotiate" in rewards
        assert rewards["direct_answer"] > rewards["negotiate"]


# ============================================================
# 顶层: DecisionEngine 测试
# ============================================================


class TestDecisionEngine:
    """决策引擎顶层编排器测试."""

    @pytest.mark.asyncio
    async def test_process_query_end_to_end(
        self,
        mock_intent_router: MagicMock,
        executor: TaskExecutor,
    ) -> None:
        """测试端到端查询处理."""
        engine = DecisionEngine(
            intent_router=mock_intent_router,
            task_executor=executor,
        )

        record = await engine.process_query("Dy3+ 的发光原理是什么")

        assert isinstance(record, ActionRecord)
        assert record.plan_id != ""
        assert record.action_type in list(ActionType)
        assert "_meta" in record.response_payload
        meta = record.response_payload["_meta"]
        assert "total_elapsed_ms" in meta
        assert "validation_score" in meta

    @pytest.mark.asyncio
    async def test_process_query_with_config(
        self,
        mock_intent_router: MagicMock,
        executor: TaskExecutor,
    ) -> None:
        """测试自定义配置."""
        config = DecisionEngineConfig(
            enable_validation=False,
            enable_feedback=False,
            enable_ucb=False,
        )
        engine = DecisionEngine(
            intent_router=mock_intent_router,
            task_executor=executor,
            config=config,
        )

        record = await engine.process_query("测试查询")
        assert record.action_type in list(ActionType)

    @pytest.mark.asyncio
    async def test_process_query_numeric_intent(
        self,
        mock_intent_router: MagicMock,
        executor: TaskExecutor,
    ) -> None:
        """测试数值意图路由."""
        mock_intent_router.route.return_value = MagicMock(
            intent=MagicMock(intent_type=MagicMock(value="numeric")),
            query="Dy3+ 的激发态波长",
            retrieval_result=MagicMock(query="Dy3+ 的激发态波长"),
        )

        engine = DecisionEngine(
            intent_router=mock_intent_router,
            task_executor=executor,
        )
        record = await engine.process_query("Dy3+ 的激发态波长")

        meta = record.response_payload["_meta"]
        assert meta["intent_type"] == "numeric"

    @pytest.mark.asyncio
    async def test_process_query_with_feedback(
        self,
        mock_intent_router: MagicMock,
        executor: TaskExecutor,
    ) -> None:
        """测试反馈闭环."""
        engine = DecisionEngine(
            intent_router=mock_intent_router,
            task_executor=executor,
            config=DecisionEngineConfig(enable_feedback=True, enable_ucb=True),
        )

        record = await engine.process_query("测试")

        # 记录正反馈
        signal = engine.record_feedback(record, rating=1.0, comment="很好")
        assert signal is not None
        assert signal.rating == 1.0

        # 记录负反馈
        signal2 = engine.record_feedback(record, rating=-1.0, feedback_type=FeedbackType.CORRECTION)
        assert signal2 is not None

    def test_record_feedback_disabled(self) -> None:
        """测试反馈禁用时返回 None."""
        config = DecisionEngineConfig(enable_feedback=False)
        engine = DecisionEngine(
            intent_router=MagicMock(),
            task_executor=MagicMock(),
            config=config,
        )
        record = ActionRecord(plan_id="p1")
        signal = engine.record_feedback(record, rating=0.5)
        assert signal is None

    def test_get_feedback_summary_with_data(self) -> None:
        """测试获取反馈摘要."""
        engine = DecisionEngine(
            intent_router=MagicMock(),
            task_executor=MagicMock(),
            config=DecisionEngineConfig(enable_feedback=True),
        )

        # 添加多条反馈
        for i in range(10):
            record = ActionRecord(plan_id=f"p{i}", action_type=ActionType.DIRECT_ANSWER)
            engine.record_feedback(record, rating=0.8, intent_type="concept")

        summary = engine.get_feedback_summary(last_hours=24)
        assert summary is not None
        assert summary["total_signals"] == 10
        assert summary["avg_rating"] == 0.8

    def test_get_feedback_summary_empty(self) -> None:
        """测试无反馈时返回 None."""
        engine = DecisionEngine(
            intent_router=MagicMock(),
            task_executor=MagicMock(),
            config=DecisionEngineConfig(enable_feedback=True),
        )
        summary = engine.get_feedback_summary(last_hours=24)
        assert summary is None

    def test_extract_intent_type_variations(self) -> None:
        """测试意图类型提取兼容性."""
        # 字典结构
        routed_dict = {"intent": {"intent_type": "numeric"}}
        assert DecisionEngine._extract_intent_type(routed_dict) == "numeric"

        # 对象结构
        routed_obj = MagicMock()
        routed_obj.intent = MagicMock()
        routed_obj.intent.intent_type = MagicMock(value="relational")
        assert DecisionEngine._extract_intent_type(routed_obj) == "relational"

        # 默认
        assert DecisionEngine._extract_intent_type({}) == "unknown"

    def test_validation_disabled_returns_default(self) -> None:
        """测试验证禁用时返回默认报告."""
        config = DecisionEngineConfig(enable_validation=False)
        engine = DecisionEngine(
            intent_router=MagicMock(),
            task_executor=MagicMock(),
            config=config,
        )

        result = ExecutionResult(plan_id="p1", status=ExecutionStatus.COMPLETED, confidence=0.8)
        report = engine._run_validation(result)
        assert report.overall_score == 0.8
        assert report.is_valid


# ============================================================
# 数据模型测试
# ============================================================


class TestDataModels:
    """数据模型行为测试."""

    def test_execution_result_compute_confidence(self) -> None:
        """测试综合置信度计算."""
        result = ExecutionResult(plan_id="p1")
        result.task_results["r1"] = TaskResult(
            task_id="r1", task_type=TaskType.REASON, confidence=0.9,
        )
        result.task_results["r2"] = TaskResult(
            task_id="r2", task_type=TaskType.RETRIEVE, confidence=0.8,
        )
        result.task_results["r3"] = TaskResult(
            task_id="r3", task_type=TaskType.SYNTHESIZE, confidence=0.7,
        )

        conf = result.compute_confidence()
        assert 0 < conf < 1
        # reason 权重最高 (0.4)，应该接近 0.9
        assert conf > 0.8

    def test_validation_report_is_valid(self) -> None:
        """测试验证报告有效性判断."""
        report = ValidationReport(overall_status=ValidationSeverity.PASS)
        assert report.is_valid

        report.overall_status = ValidationSeverity.ERROR
        assert not report.is_valid

        report.overall_status = ValidationSeverity.CRITICAL
        assert not report.is_valid

    def test_validation_report_needs_human_review(self) -> None:
        """测试人工复核判断."""
        report = ValidationReport(overall_status=ValidationSeverity.WARNING)
        assert report.needs_human_review

        report.overall_status = ValidationSeverity.ERROR
        assert report.needs_human_review

        report.overall_status = ValidationSeverity.PASS
        assert not report.needs_human_review

    def test_task_result_success_failed(self) -> None:
        """测试任务结果状态判断."""
        success = TaskResult(task_id="t1", task_type=TaskType.RETRIEVE, status=ExecutionStatus.COMPLETED)
        assert success.is_success
        assert not success.is_failed

        failed = TaskResult(task_id="t2", task_type=TaskType.REASON, status=ExecutionStatus.FAILED)
        assert not failed.is_success
        assert failed.is_failed

        timeout = TaskResult(task_id="t3", task_type=TaskType.VERIFY, status=ExecutionStatus.TIMEOUT)
        assert timeout.is_failed

    def test_resource_budget_within_budget(self) -> None:
        """测试资源预算检查."""
        budget = ResourceBudget(max_latency_ms=100, max_tool_calls=5)
        assert budget.is_within_budget(elapsed_ms=50, tool_calls=3)
        assert not budget.is_within_budget(elapsed_ms=150, tool_calls=3)
        assert not budget.is_within_budget(elapsed_ms=50, tool_calls=6)

    def test_decision_plan_get_task(self) -> None:
        """测试按 ID 获取子任务."""
        plan = DecisionPlan(
            sub_tasks=[
                SubTask(task_id="t1", task_type=TaskType.RETRIEVE),
                SubTask(task_id="t2", task_type=TaskType.REASON),
            ]
        )
        assert plan.get_task("t1") is not None
        assert plan.get_task("t2") is not None
        assert plan.get_task("t3") is None

    def test_decision_plan_get_results_by_type(self) -> None:
        """测试按类型获取结果."""
        result = ExecutionResult(plan_id="p1")
        result.task_results["t1"] = TaskResult(task_id="t1", task_type=TaskType.RETRIEVE)
        result.task_results["t2"] = TaskResult(task_id="t2", task_type=TaskType.RETRIEVE)
        result.task_results["t3"] = TaskResult(task_id="t3", task_type=TaskType.REASON)

        retrieve_results = result.get_results_by_type(TaskType.RETRIEVE)
        assert len(retrieve_results) == 2


# ============================================================
# 集成测试
# ============================================================


class TestIntegration:
    """组件间集成测试."""

    def test_t4_to_t5_flow(self, sample_execution_result: ExecutionResult) -> None:
        """测试 T4 -> T5 数据流."""
        # T4: 验证
        validator = ValidationOrchestrator()
        report = validator.validate(sample_execution_result)

        # T5: 行动选择
        selector = ActionSelector(use_ucb=False)
        record = selector.select(report, sample_execution_result)

        assert record.validation_score == report.overall_score
        assert record.execution_confidence == sample_execution_result.confidence
        assert record.plan_id == sample_execution_result.plan_id

    def test_t5_to_t6_flow(self) -> None:
        """测试 T5 -> T6 数据流."""
        # T5: 产生 ActionRecord
        record = ActionRecord(
            plan_id="p1",
            action_type=ActionType.DIRECT_ANSWER,
            validation_score=0.9,
            execution_confidence=0.85,
        )

        # T6: 反馈聚合
        aggregator = FeedbackAggregator()
        signal = aggregator.add_explicit_feedback(record, rating=1.0, comment="满意")

        assert signal.plan_id == record.plan_id
        assert signal.action_type == record.action_type.value
        assert signal.validation_score == record.validation_score

    def test_ucb_learning_cycle(self) -> None:
        """测试 UCB 学习闭环."""
        selector = ActionSelector(use_ucb=True)

        # 模拟多次反馈
        for _ in range(5):
            selector.feedback(ActionType.DIRECT_ANSWER, 1.0)
        for _ in range(5):
            selector.feedback(ActionType.HUMAN_CONFIRM, -0.5)

        stats = selector.get_ucb_stats()
        assert stats["direct_answer"]["avg_reward"] > stats["human_confirm"]["avg_reward"]

    def test_full_pipeline_without_execution(self) -> None:
        """测试除 T3 外的完整管道（纯数据流）."""
        # T2: 生成计划
        planner = DecisionPlanner()
        routed = {"intent": {"intent_type": "concept"}, "query": "测试"}
        plan = planner.plan(routed)

        # 构造模拟 T3 结果
        result = ExecutionResult(plan_id=plan.plan_id, status=ExecutionStatus.COMPLETED)
        result.task_results["retrieve_concept"] = TaskResult(
            task_id="retrieve_concept", task_type=TaskType.RETRIEVE,
            status=ExecutionStatus.COMPLETED, confidence=0.9,
        )
        result.task_results["synthesize_concept"] = TaskResult(
            task_id="synthesize_concept", task_type=TaskType.SYNTHESIZE,
            status=ExecutionStatus.COMPLETED, confidence=0.85,
        )
        result.confidence = 0.87

        # T4
        validator = ValidationOrchestrator()
        report = validator.validate(result)

        # T5
        selector = ActionSelector(use_ucb=False)
        record = selector.select(report, result)

        # T6
        aggregator = FeedbackAggregator()
        aggregator.add_explicit_feedback(record, rating=0.8)

        assert record.action_type == ActionType.DIRECT_ANSWER
        assert aggregator._signals[0].rating == 0.8
