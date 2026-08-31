"""默认 Agent 运行时与不确定确认流程测试."""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l4.models import ActionType
from dy3_polaris.l5.agent_workers import (
    DIAGNOSIS_AGENT_ID,
    GENERATION_AGENT_ID,
    GUIDANCE_AGENT_ID,
    REVIEW_AGENT_ID,
    _detect_ambiguity,
)
from dy3_polaris.l5.default_agents import (
    DECISION_AGENT_ID,
    build_default_agent_runtime,
    build_default_agents,
    build_default_prompt_manager,
)
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5 import unified_app as unified_app_runtime
from dy3_polaris.l5.unified_app import (
    UnifiedApp,
    _P0_RESPONSE_FIELD_PRODUCERS,
    _P0_RESPONSE_FORBIDDEN_V2_FIELDS,
    _guard_query_response,
)
from tests.l5.test_task_understanding import _stub_current_workers


class TestDefaultAgentRuntime:
    """四个核心 Agent 的默认注册与实例化."""

    def test_four_core_agents_are_registered(self) -> None:
        definitions = build_default_agents()
        ids = {definition.id for definition in definitions}
        assert ids == {
            "agent.learning.diagnosis",
            "agent.knowledge.generation",
            "agent.quality.review",
            DECISION_AGENT_ID,
        }

    def test_decision_agent_has_full_authority(self) -> None:
        runtime = build_default_agent_runtime()
        decision = runtime.registry.get(DECISION_AGENT_ID)
        assert decision is not None
        assert decision.decision_authority.scheduling is True
        assert decision.decision_authority.intervention is True
        assert decision.decision_authority.adaptive is True
        assert decision.self_evolution.enabled is True

    def test_prompt_manager_has_all_default_templates(self) -> None:
        manager = build_default_prompt_manager()
        for template_id in (
            "tpl.diagnosis",
            "tpl.generation",
            "tpl.review",
            "tpl.guidance",
        ):
            assert manager.get_active(template_id) is not None

    @pytest.mark.asyncio
    async def test_decision_agent_can_instantiate(self) -> None:
        runtime = build_default_agent_runtime()
        await runtime.ensure_instances([DECISION_AGENT_ID])
        instance = runtime.get_instance(DECISION_AGENT_ID)
        assert instance is not None
        assert instance.state.value == "active"
        assert instance.health_check()["healthy"] is True

    def test_full_app_exposes_agents_endpoint(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())
        data = client.get("/l5/agents").json()["data"]
        assert data["total"] == 4
        assert data["decision_agent"] == DECISION_AGENT_ID
        assert any(
            agent["id"] == DECISION_AGENT_ID for agent in data["agents"]
        )

    @pytest.mark.asyncio
    async def test_all_agents_can_run_without_dependencies(self) -> None:
        runtime = build_default_agent_runtime()
        for agent_id in (
            DIAGNOSIS_AGENT_ID,
            GENERATION_AGENT_ID,
            REVIEW_AGENT_ID,
            GUIDANCE_AGENT_ID,
        ):
            result = await runtime.run(
                agent_id,
                {
                    "learner_id": "demo-learner",
                    "query": "Dy3+ 的量子效率受哪些因素影响？",
                    "content": "Dy3+ 离子的发射波长为 575nm",
                },
            )
            assert result["agent_id"] == agent_id
            assert result["status"] == "completed"

    def test_agent_run_endpoints_work(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())
        login = client.post(
            "/l1/api/v1/auth/login",
            json={"student_id": "DY20248888", "password": "admin888"},
        )
        headers = {
            "Authorization": "Bearer "
            + login.json()["data"]["access_token"]
        }

        diagnosis = client.post(
            f"/l5/agents/{DIAGNOSIS_AGENT_ID}/run",
            json={"learner_id": "demo-learner"},
            headers=headers,
        )
        assert diagnosis.status_code == 200
        assert diagnosis.json()["data"]["agent_id"] == DIAGNOSIS_AGENT_ID

        generation = client.post(
            f"/l5/agents/{GENERATION_AGENT_ID}/run",
            json={"query": "Dy3+ 的量子效率受哪些因素影响？"},
            headers=headers,
        )
        assert generation.status_code == 200
        assert generation.json()["data"]["answer"]

        review = client.post(
            f"/l5/agents/{REVIEW_AGENT_ID}/run",
            json={"content": "Dy3+ 离子的发射波长为 575nm"},
            headers=headers,
        )
        assert review.status_code == 200
        assert review.json()["data"]["verdict"] in (
            "approved",
            "needs_review",
            "rejected",
        )

        guidance = client.post(
            f"/l5/agents/{GUIDANCE_AGENT_ID}/run",
            json={
                "learner_id": "demo-learner",
                "query": "Dy3+ 的量子效率受哪些因素影响？",
            },
            headers=headers,
        )
        assert guidance.status_code == 200
        guidance_data = guidance.json()["data"]
        assert len(guidance_data["pipeline"]) == 3

    def test_agent_run_requires_l1_auth(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())
        resp = client.post(
            f"/l5/agents/{DIAGNOSIS_AGENT_ID}/run",
            json={"learner_id": "demo-learner"},
        )
        assert resp.status_code == 401

    def test_agent_run_enforces_l1_permission(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())
        login = client.post(
            "/l1/api/v1/auth/login",
            json={"student_id": "DY20240001", "password": "demo123"},
        )
        headers = {
            "Authorization": "Bearer "
            + login.json()["data"]["access_token"]
        }
        # 本科生没有审核 Agent 权限
        resp = client.post(
            f"/l5/agents/{REVIEW_AGENT_ID}/run",
            json={"content": "待审核内容"},
            headers=headers,
        )
        assert resp.status_code == 403


class _FakeDecisionEngine:
    """固定返回 NEGOTIATE 结果的假决策引擎."""

    async def process_query(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            action_type=ActionType.NEGOTIATE,
            confidence=0.42,
            plan_id="plan-fake-1",
            selection_reason="验证有警告",
            validation_score=0.42,
            clarification_questions=["不同来源说法不同，您倾向参考哪个来源？"],
            response_payload={
                "_meta": {
                    "intent_type": "concept",
                    "total_elapsed_ms": 12,
                    "validation_score": 0.42,
                    "retry_count": 1,
                    "safety_level": "safe",
                },
                "answers": [{"text": "候选答案，等待确认"}],
                "evidence": [],
                "escalation_reason": ["证据有限，存在不确定性"],
            },
        )


class _SuccessfulFallbackDecisionEngine:
    """返回高置信度答案，用于验证 fallback 不伪造 Task COMPLETED。"""

    async def process_query(self, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            action_type=ActionType.DIRECT_ANSWER,
            confidence=0.9,
            plan_id="plan-fallback-success",
            selection_reason="L4 fallback produced an answer",
            validation_score=0.9,
            clarification_questions=[],
            response_payload={
                "_meta": {
                    "intent_type": "concept",
                    "total_elapsed_ms": 12,
                    "validation_score": 0.9,
                    "retry_count": 0,
                    "safety_level": "safe",
                },
                "answers": [{"text": "L4 fallback answer"}],
                "evidence": [],
            },
        )


class _CapturingGuidanceRuntime:
    """最小运行时替身：验证 /api/query 传入的任务身份。"""

    def __init__(self) -> None:
        self.input_data: dict[str, object] | None = None

    async def run(self, agent_id: str, input_data: dict[str, object]) -> dict[str, object]:
        assert agent_id == GUIDANCE_AGENT_ID
        self.input_data = input_data
        return {
            "agent_id": GUIDANCE_AGENT_ID,
            "status": "completed",
            "task_id": input_data["task_id"],
            "answer": "正常回答",
            "confidence": 0.9,
            "review": {
                "agent_id": "agent.quality.review",
                "status": "completed",
                "verdict": "approved",
            },
            "evidence": [{"content": "运行时证据", "source": "runtime"}],
            "recommended_path": [{"step": "运行时建议路径"}],
            "action_type": "answer",
            "quality_release": {
                "status": "FULL_RELEASE",
                "eligible": True,
                "message": "reviewed answer released",
                "reason_codes": [],
                "review_status": "completed",
                "review_verdict": "approved",
                "correction_count": 0,
                "evidence_versions": [1],
            },
        }


class _FailingGuidanceRuntime:
    """验证 Agent 异常后 API 回退不受任务身份字段影响。"""

    async def run(self, agent_id: str, input_data: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("forced guidance failure")


class _ReviewingGuidanceRuntime:
    """返回答案但停留在 REVIEWING，用于验证完成门禁。"""

    async def run(self, agent_id: str, input_data: dict[str, object]) -> dict[str, object]:
        task_context = input_data["task_context"]
        for state in ("RETRIEVING", "COLLABORATING", "REVIEWING"):
            task_state_runtime.set_task_state(task_context, state)
        return {
            "agent_id": GUIDANCE_AGENT_ID,
            "status": "completed",
            "task_id": input_data["task_id"],
            "answer": "审核尚未结束的候选回答",
            "confidence": 0.9,
            "review": {
                "agent_id": "agent.quality.review",
                "status": "completed",
                "verdict": "needs_review",
            },
            "evidence": [],
            "recommended_path": [],
            "quality_release": {
                "status": "WITHHOLD",
                "eligible": False,
                "message": "review is unresolved",
                "reason_codes": ["review_not_approved"],
                "review_status": "completed",
                "review_verdict": "needs_review",
                "correction_count": 0,
                "evidence_versions": [],
            },
        }


class _AskUserGuidanceRuntime:
    """真实 Agent 路径的 ASK_USER 门控，不使用 L4 兼容回退。"""

    async def run(self, agent_id: str, input_data: dict[str, object]) -> dict[str, object]:
        task_context = input_data["task_context"]
        for state in ("RETRIEVING", "COLLABORATING", "REVIEWING"):
            task_state_runtime.set_task_state(task_context, state)
        return {
            "agent_id": GUIDANCE_AGENT_ID,
            "status": "completed",
            "task_id": input_data["task_id"],
            "answer": "不得由确认接口释放的候选草稿",
            "confidence": 0.42,
            "review": {
                "agent_id": "agent.quality.review",
                "status": "completed",
                "verdict": "needs_review",
                "reason": "需要补充评价目标",
            },
            "evidence": [],
            "recommended_path": [],
            "requires_confirmation": True,
            "clarify": {
                "type": "decision_clarification",
                "message": "请明确评价目标",
                "options": ["发光效率", "健康照明风险"],
            },
            "action_type": "clarify",
            "quality_release": {
                "status": "ASK_USER",
                "eligible": False,
                "message": "clarification required",
                "reason_codes": ["clarification_required"],
                "review_status": "completed",
                "review_verdict": "needs_review",
                "correction_count": 0,
                "evidence_versions": [],
            },
        }


class TestUncertaintyConfirmation:
    """不确定结果先向提问者确认，确认后才返回答案."""

    def test_uncertain_query_requires_confirmation(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = _AskUserGuidanceRuntime()
        client = TestClient(builder.create_app())

        resp = client.post(
            "/api/query",
            json={"query": "不确定的问题", "learner_id": "demo-learner"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["requires_confirmation"] is True
        assert data["answer"] == ""
        assert data["plan_id"].startswith("agent-guidance-")
        assert data["task_id"].startswith("task-")
        assert data["task_id"] != data["plan_id"]
        assert data["task_state"] != "COMPLETED"
        assert data["confirmation_questions"]
        pending = builder._handlers._confirmations.get(data["plan_id"])
        assert pending is not None
        assert pending.task_id == data["task_id"]
        assert pending.task_state == data["task_state"]
        assert pending.task_events == data["task_events"]
        assert not any(
            event["state_after"] == "COMPLETED"
            for event in data["task_events"]
        )

        confirmed = client.post(
            "/api/query/confirm",
            json={"plan_id": data["plan_id"], "decision": "accept"},
        )
        assert confirmed.status_code == 200
        confirm_data = confirmed.json()["data"]
        assert confirm_data["confirmed"] is True
        assert confirm_data["answer"] == ""
        assert confirm_data["action_type"] == "clarify"
        assert confirm_data["task_id"] == data["task_id"]
        assert confirm_data["task_state"] == data["task_state"]
        assert confirm_data["task_events"] == data["task_events"]

    def test_rejected_confirmation_is_consumed(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = _AskUserGuidanceRuntime()
        client = TestClient(builder.create_app())

        initial = client.post(
            "/api/query",
            json={"query": "不确定的问题", "learner_id": "demo-learner"},
        )
        initial_data = initial.json()["data"]
        rejected = client.post(
            "/api/query/confirm",
            json={"plan_id": initial_data["plan_id"], "decision": "reject"},
        )
        rejected_data = rejected.json()["data"]
        assert rejected_data["confirmed"] is False
        assert rejected_data["task_id"] == initial_data["task_id"]
        assert rejected_data["task_state"] == initial_data["task_state"]
        assert rejected_data["task_events"] == initial_data["task_events"]
        assert rejected_data["task_state"] != "COMPLETED"

        again = client.post(
            "/api/query/confirm",
            json={"plan_id": initial_data["plan_id"], "decision": "accept"},
        )
        assert again.status_code == 404


class TestTaskIdentity:
    """P0-01: 最小 Task Identity 贯穿 API、运行时和确认路径。"""

    @pytest.mark.asyncio
    async def test_guidance_worker_preserves_task_id(self) -> None:
        runtime = build_default_agent_runtime()
        result = await runtime.run(
            GUIDANCE_AGENT_ID,
            {
                "learner_id": "demo-learner",
                "query": "Dy3+ 的量子效率受哪些因素影响？",
                "task_id": "task-runtime-chain-001",
            },
        )
        assert result["task_id"] == "task-runtime-chain-001"

    def test_api_query_returns_and_passes_same_task_id(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        runtime = _CapturingGuidanceRuntime()
        builder._handlers._agents = runtime
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "正常问题"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["task_id"].startswith("task-")
        assert runtime.input_data is not None
        assert runtime.input_data["task_id"] == data["task_id"]
        assert data["answer"] == "正常回答"
        for field in ("confidence", "review", "evidence", "recommended_path", "action_type", "plan_id"):
            assert field in data

    def test_api_query_default_runtime_reaches_guidance_with_same_task_id(self) -> None:
        """验收真实主链，而非只验证运行时替身的输入。"""
        builder = UnifiedApp.create_full_app_builder()
        runtime = builder._handlers._agents
        assert runtime is not None
        guidance_worker = runtime._workers[GUIDANCE_AGENT_ID]
        guidance_inputs: list[dict[str, object]] = []

        async def observing_guidance_worker(input_data: dict[str, object]) -> dict[str, object]:
            guidance_inputs.append(dict(input_data))
            return await guidance_worker(input_data)

        runtime._workers[GUIDANCE_AGENT_ID] = observing_guidance_worker
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "Dy3+ 的量子效率受哪些因素影响？"})

        assert response.status_code == 200
        task_id = response.json()["data"]["task_id"]
        assert task_id.startswith("task-")
        assert len(guidance_inputs) == 1
        assert guidance_inputs[0]["task_id"] == task_id

    def test_api_query_writes_correlated_agent_audit_records(self, tmp_path) -> None:
        """真实 /api/query 主链写入可按 trace/session/task 追踪的 Agent 审计."""
        builder = UnifiedApp.create_full_app_builder(data_dir=str(tmp_path))
        client = TestClient(builder.create_app())

        response = client.post(
            "/api/query",
            json={
                "query": "Dy3+ 的量子效率受哪些因素影响？",
                "learner_id": "audit-learner",
            },
        )

        assert response.status_code == 200
        task_id = response.json()["data"]["task_id"]
        trace_id = response.headers["x-trace-id"]
        audit_engine = builder._governance_router._subsys.audit_engine
        logs = audit_engine.query(trace_id=trace_id, limit=50)
        agent_logs = [log for log in logs if log.action == "agent_invoke"]

        assert trace_id.startswith("tr-")
        assert agent_logs
        assert all(log.trace_id == trace_id for log in agent_logs)
        assert all(log.session_id for log in agent_logs)
        assert all(log.input_context.get("task_id") == task_id for log in agent_logs)
        assert all(log.output_result.get("agent_id") for log in agent_logs)
        assert any(log.latency_ms > 0 for log in agent_logs)

    def test_two_api_queries_get_distinct_task_ids(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        runtime = _CapturingGuidanceRuntime()
        builder._handlers._agents = runtime
        client = TestClient(builder.create_app())

        first = client.post("/api/query", json={"query": "第一个正常问题"})
        second = client.post("/api/query", json={"query": "第二个正常问题"})

        first_task_id = first.json()["data"]["task_id"]
        second_task_id = second.json()["data"]["task_id"]
        assert first_task_id.startswith("task-")
        assert second_task_id.startswith("task-")
        assert first_task_id != second_task_id

    def test_agent_failure_keeps_task_id_in_response(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = _FailingGuidanceRuntime()
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "Agent 失败回退"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["task_id"].startswith("task-")
        assert data["task_state"] != "COMPLETED"
        assert not any(
            event["state_after"] == "COMPLETED"
            for event in data["task_events"]
        )


class TestMinimalTaskState:
    """P0-02: 任务级状态只跟随当前真实主链阶段。"""

    def test_successful_query_has_monotonic_real_state_lifecycle(self, monkeypatch) -> None:
        _stub_current_workers(monkeypatch, [])
        observed: list[tuple[str, str]] = []
        original_set_task_state = task_state_runtime.set_task_state

        def observe_state(task_context, state: str, *args, **kwargs) -> str:
            result = original_set_task_state(task_context, state, *args, **kwargs)
            observed.append((task_context["task_id"], result))
            return result

        monkeypatch.setattr(task_state_runtime, "set_task_state", observe_state)
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())

        response = client.post(
            "/api/query",
            json={"query": "Dy3+ 的量子效率受哪些因素影响？"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["task_state"] == "COMPLETED"
        assert {task_id for task_id, _ in observed} == {data["task_id"]}
        states = [state for _, state in observed]
        assert states == [
            "UNDERSTANDING",
            "PLANNING",
            "RETRIEVING",
            "COLLABORATING",
            "REVIEWING",
            "ANSWERING",
            "COMPLETED",
        ]
        state_indexes = [task_state_runtime.P0_TASK_STATES.index(state) for state in states]
        assert state_indexes == sorted(state_indexes)

    def test_minimal_state_rejects_unknown_and_backward_states(self) -> None:
        task_context = task_state_runtime.create_task_context("task-state-test")
        assert task_context["state"] == "CREATED"
        task_state_runtime.set_task_state(task_context, "UNDERSTANDING")

        with pytest.raises(ValueError, match="unsupported task state"):
            task_state_runtime.set_task_state(task_context, "REASONING")
        with pytest.raises(ValueError, match="cannot transition"):
            task_state_runtime.set_task_state(task_context, "CREATED")

    def test_task_state_is_additive_to_existing_response(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        runtime = _CapturingGuidanceRuntime()
        builder._handlers._agents = runtime
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "兼容性检查"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert "task_state" in data
        assert data["task_state"] == "UNDERSTANDING"
        assert "task_events" in data
        for field in (
            "task_id",
            "answer",
            "confidence",
            "review",
            "evidence",
            "action_type",
            "recommended_path",
            "plan_id",
        ):
            assert field in data

    def test_reviewing_with_answer_cannot_complete(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = _ReviewingGuidanceRuntime()
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "审核未完成检查"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["answer"] == ""
        assert data["quality_release"]["status"] == "WITHHOLD"
        assert data["task_state"] == "REVIEWING"


class TestMinimalTaskEvents:
    """P0-03: TaskEvent 由真实状态和 Agent 执行边界即时产生。"""

    def test_successful_query_emits_ordered_real_task_events(self, monkeypatch) -> None:
        _stub_current_workers(monkeypatch, [])
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())

        response = client.post(
            "/api/query",
            json={"query": "Dy3+ 的量子效率受哪些因素影响？"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        events = data["task_events"]
        assert events
        assert {event["task_id"] for event in events} == {data["task_id"]}
        assert [event["timestamp"] for event in events] == sorted(
            event["timestamp"] for event in events
        )
        assert len({event["event_id"] for event in events}) == len(events)
        assert {event["event_type"] for event in events} >= {
            "StateChanged",
            "AgentStarted",
            "AgentFinished",
            "ReviewCompleted",
        }

        state_changes = [
            (event["state_before"], event["state_after"])
            for event in events
            if event["event_type"] == "StateChanged"
        ]
        assert state_changes == [
            ("", "CREATED"),
            ("CREATED", "UNDERSTANDING"),
            ("UNDERSTANDING", "PLANNING"),
            ("PLANNING", "RETRIEVING"),
            ("RETRIEVING", "COLLABORATING"),
            ("COLLABORATING", "REVIEWING"),
            ("REVIEWING", "ANSWERING"),
            ("ANSWERING", "COMPLETED"),
        ]
        assert events[-1]["event_type"] == "TaskCompleted"
        assert events[-1]["state_before"] == "COMPLETED"
        assert events[-1]["state_after"] == "COMPLETED"

        for event in events:
            if event["event_type"] != "StateChanged":
                assert event["state_before"] == event["state_after"]

        review_completed_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "ReviewCompleted"
        )
        answering_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "StateChanged"
            and event["state_after"] == "ANSWERING"
        )
        assert review_completed_index < answering_index
        assert events[review_completed_index]["producer"] == REVIEW_AGENT_ID
        for agent_id in (
            DIAGNOSIS_AGENT_ID,
            GENERATION_AGENT_ID,
            REVIEW_AGENT_ID,
            GUIDANCE_AGENT_ID,
        ):
            started_index = next(
                index
                for index, event in enumerate(events)
                if event["event_type"] == "AgentStarted"
                and event["producer"] == agent_id
            )
            finished_index = next(
                index
                for index, event in enumerate(events)
                if event["event_type"] == "AgentFinished"
                and event["producer"] == agent_id
            )
            assert started_index < finished_index

    def test_event_schema_and_state_producers_are_explicit(self) -> None:
        task_context = task_state_runtime.create_task_context(
            "task-event-schema",
            producer="api_query",
        )
        task_state_runtime.set_task_state(
            task_context,
            "UNDERSTANDING",
            producer="api_query",
        )
        task_state_runtime.record_task_event(
            task_context,
            "AgentStarted",
            DIAGNOSIS_AGENT_ID,
        )

        events = task_state_runtime.get_task_events(task_context)
        assert [event["producer"] for event in events[:2]] == [
            "api_query",
            "api_query",
        ]
        required_fields = {
            "event_id",
            "task_id",
            "event_kind",
            "event_type",
            "timestamp",
            "producer",
            "state_before",
            "state_after",
        }
        assert all(required_fields <= set(event) for event in events)
        with pytest.raises(ValueError, match="unsupported task event type"):
            task_state_runtime.record_task_event(
                task_context,
                "UnknownEvent",
                "test",
            )


class TestP0ResponseGuardrail:
    """P0-04: 当前响应字段与任务事实、真实 producer 保持一致。"""

    def test_success_response_keeps_current_fields_without_fake_v2_fields(self, monkeypatch) -> None:
        _stub_current_workers(monkeypatch, [])
        builder = UnifiedApp.create_full_app_builder()
        client = TestClient(builder.create_app())

        response = client.post(
            "/api/query",
            json={"query": "Dy3+ 的量子效率受哪些因素影响？"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["task_id"].startswith("task-")
        assert data["task_state"] == "COMPLETED"
        assert data["answer"]
        for field in (
            "evidence",
            "review",
            "confidence",
            "action_type",
            "recommended_path",
        ):
            assert field in data
        assert not set(_P0_RESPONSE_FORBIDDEN_V2_FIELDS) & set(data)

    def test_guard_uses_task_context_as_identity_and_state_source(self) -> None:
        task_context = task_state_runtime.create_task_context("task-guard-source")
        task_state_runtime.set_task_state(task_context, "UNDERSTANDING")
        response_data = {
            "task_id": "agent-spoofed-task-id",
            "task_state": "COMPLETED",
            "task_events": task_state_runtime.get_task_events(task_context),
            "answer": "当前真实回答",
            "evidence": [],
            "review": {},
            "confidence": 0.5,
            "action_type": "answer",
            "recommended_path": [],
        }

        guarded = _guard_query_response(response_data, task_context)

        assert guarded["task_id"] == "task-guard-source"
        assert guarded["task_state"] == "UNDERSTANDING"

    @pytest.mark.parametrize("field", _P0_RESPONSE_FORBIDDEN_V2_FIELDS)
    def test_guard_rejects_unimplemented_v2_fields(self, field: str) -> None:
        task_context = task_state_runtime.create_task_context("task-no-fake-v2")
        response_data = {
            "task_id": "task-no-fake-v2",
            "task_state": "CREATED",
            "task_events": task_state_runtime.get_task_events(task_context),
            "answer": "",
            "evidence": [],
            "review": {},
            "confidence": 0.0,
            "action_type": "",
            "recommended_path": [],
            field: {},
        }

        with pytest.raises(RuntimeError, match="unimplemented V2 fields"):
            _guard_query_response(response_data, task_context)

    def test_current_field_producer_mapping_is_complete(self) -> None:
        assert set(_P0_RESPONSE_FIELD_PRODUCERS) == {
            "answer",
            "review",
            "confidence",
            "evidence",
            "recommended_path",
        }
        assert all(_P0_RESPONSE_FIELD_PRODUCERS.values())

    def test_current_agent_fields_pass_through_without_semantic_rewrite(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = _CapturingGuidanceRuntime()
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "字段来源检查"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["answer"] == "正常回答"
        assert data["review"] == {
            "agent_id": "agent.quality.review",
            "status": "completed",
            "verdict": "approved",
        }
        assert data["confidence"] == 0.9
        assert data["evidence"] == [
            {"content": "运行时证据", "source": "runtime"}
        ]
        assert data["recommended_path"] == [{"step": "运行时建议路径"}]
        assert data["action_type"] == "answer"


class TestP0VerificationFailurePaths:
    """P0-05: exception、timeout、fallback 均不能伪造 COMPLETED。"""

    def test_timeout_fallback_keeps_real_noncompleted_state(self, monkeypatch) -> None:
        async def force_timeout(awaitable, *, timeout):
            assert timeout == 25.0
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError("forced timeout")

        monkeypatch.setattr(unified_app_runtime.asyncio, "wait_for", force_timeout)
        builder = UnifiedApp.create_full_app_builder()
        builder.bridge.decision_engine = _SuccessfulFallbackDecisionEngine()
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "timeout gate"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["answer"] == ""
        assert data["quality_release"]["status"] == "DEGRADED"
        assert data["task_state"] == "PARTIAL"
        assert data["task_events"][-1]["state_after"] == "PARTIAL"
        assert not any(
            event["state_after"] == "COMPLETED"
            for event in data["task_events"]
        )

    def test_direct_l4_fallback_keeps_real_noncompleted_state(self) -> None:
        builder = UnifiedApp.create_full_app_builder()
        builder._handlers._agents = None
        builder.bridge.decision_engine = _SuccessfulFallbackDecisionEngine()
        client = TestClient(builder.create_app())

        response = client.post("/api/query", json={"query": "fallback gate"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["answer"] == ""
        assert data["quality_release"]["status"] == "DEGRADED"
        assert data["task_state"] == "PARTIAL"
        assert not any(
            event["state_after"] == "COMPLETED"
            for event in data["task_events"]
        )


class TestAmbiguityDetection:
    """模糊检测: 定义类提问不应误判为模糊 (回归 dy是什么 被误澄清)."""

    @pytest.mark.parametrize(
        "query",
        [
            "dy是什么",
            "什么是镝",
            "Dy3+是什么",
            "镝是什么元素",
            "dy是啥",
        ],
    )
    def test_definition_queries_not_ambiguous(self, query: str) -> None:
        assert _detect_ambiguity(query) is None

    @pytest.mark.parametrize("query", ["dy", "镝", "er", "铒"])
    def test_bare_element_still_clarifies(self, query: str) -> None:
        assert _detect_ambiguity(query) is not None
