"""统一应用组装器 — 将所有层 Router 组装为单一 Starlette 应用.

融合世界先进方案的统一应用架构:
- Knewton SOA: 所有服务通过单一入口暴露, 前缀路由隔离
- Duolingo EDA: 统一健康检查 + 事件总线集成
- LangGraph: 图节点组装 + 统一状态管理
- Temporal: Workflow/Activity 统一入口 + API 发现
- Google ADK: Session/State/Memory 统一管理入口

核心能力:
1. 多层 Router 挂载: L2/L3/L4/L5/L6 各层 API 通过 Mount 组装
2. 统一健康检查: GET /health 聚合所有层状态
3. API 发现端点: GET /api/info 列出所有可用端点
4. CORS 中间件: 统一跨域配置
5. 集成桥接: 内置 IntegrationBridge 实现跨层通信

端点列表:
- GET  /health:       统一健康检查 (聚合所有层)
- GET  /api/info:     API 发现 (所有可用端点)
- /l2/*:              L2 个性化层路由
- /l3/*:              L3 知识层路由 (可选)
- /l4/*:              L4 决策引擎路由
- /l5/*:              L5 Agent Runtime 路由
- /l6/*:              L6 协议基础设施路由 (可选)

使用示例::

    from dy3_polaris.l5.unified_app import UnifiedApp

    app_builder = UnifiedApp(
        irt_service=irt_service,
        profile_service=profile_service,
        memory_service=memory_service,
        decision_engine=decision_engine,
        orchestration_engine=orchestration_engine,
        session_manager=session_manager,
        message_bus=message_bus,
    )
    app = app_builder.create_app()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from copy import deepcopy
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

# 请求级追踪 (contextvars trace_id 中间件)
from dy3_polaris.l5.tracing import TraceIDMiddleware, get_trace_id
# 统一安全网关 (写端点鉴权, 复用 L1 JWT)
from dy3_polaris.l5.security_gateway import SecurityGatewayMiddleware
# 幂等键中间件 (X-Idempotency-Key 去重)
from dy3_polaris.l5.idempotency import IdempotencyMiddleware

from dy3_polaris.l2.api.router import L2Router
from dy3_polaris.l4.api.router import L4Router
from dy3_polaris.l5.api.router import L5Router
from dy3_polaris.l5.confirmation import (
    ConfirmationStore,
    PendingConfirmation,
    extract_answer,
)
from dy3_polaris.l5.integration_bridge import IntegrationBridge
from dy3_polaris.l5 import task_state as task_state_runtime
from dy3_polaris.l5.learning_resources import build_resource_interaction_event
from dy3_polaris.l5.learning_workspace import (
    build_learning_workspace_view,
    public_learning_workspace_projection,
)
from dy3_polaris.l5.teaching_memory import (
    PracticeValidationEvent,
    commit_practice_validation,
    commit_resource_interaction,
)
from dy3_polaris.l5.viz_generator import generate_for_api as _viz_generate

_logger = logging.getLogger("dy3_polaris.l5.unified_app")


def _l3_snapshot_dir() -> Path:
    """Active domain package snapshot root (project asset, never ``/tmp``)."""
    from dy3_polaris.l3.domain_package import active_domain_package

    return active_domain_package().snapshot_root


def _l3_snapshot_path() -> Path:
    """Exact reviewed snapshot selected by the active domain package."""
    from dy3_polaris.l3.domain_package import active_domain_package

    return active_domain_package().snapshot_path

_CONFIRM_ACTIONS = {"negotiate", "human_confirm"}
_CONFIRM_CONFIDENCE_THRESHOLD = 0.55


# ============================================================
# 统一响应
# ============================================================


# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import ok as _ok


# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err


def _dump(obj: Any) -> Any:
    """宽松序列化: 优先 to_dict/model_dump, 否则遍历 __dict__."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, dict):
        return {str(k): _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _dump(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _assert_public_dto(value: Any, path: str = "response") -> None:
    """Fail closed when a response contains a runtime/private Python object.

    Public endpoints must build an explicit plain-data projection.  This guard
    intentionally does not call ``model_dump``, ``to_dict`` or ``__dict__`` on
    an unknown object because doing so could silently expose private carrier,
    learner-memory, or request-local contract state.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"public DTO key must be str at {path}")
            _assert_public_dto(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_public_dto(item, f"{path}[{index}]")
        return
    raise RuntimeError(
        f"public DTO contains non-serializable runtime object at {path}: "
        f"{type(value).__module__}.{type(value).__name__}"
    )


def _l1_session_dict(session: Any) -> dict[str, Any]:
    """L1 用户会话 → 前端可见字典 (统一会话入口, 含聚合执行记录)."""
    agent_sessions = list(getattr(session, "agent_sessions", []) or [])
    return {
        "session_id": getattr(session, "session_id", ""),
        "session_type": getattr(session, "session_type", "query"),
        "status": getattr(session, "status", "active"),
        "created_at": getattr(session, "created_at", 0),
        "updated_at": getattr(session, "updated_at", 0),
        "question_count": int(getattr(session, "question_count", 0) or 0),
        "agent_sessions": agent_sessions,
        "agent_execution_count": len(agent_sessions),
        "unified": True,
    }


# P0-04 只追踪当前真实字段来源；这些内部常量不扩展 API schema。
_P0_RESPONSE_REQUIRED_FIELDS = (
    "task_id",
    "task_state",
    "answer",
    "evidence",
    "review",
    "confidence",
    "action_type",
    "recommended_path",
)
_P0_RESPONSE_FORBIDDEN_V2_FIELDS = (
    "facts",
    "reasoning",
    "recommendations",
    "provenance",
    "learner_update",
    "next_action",
)
_P0_RESPONSE_FIELD_PRODUCERS = {
    "answer": (
        "agent.guidance.decision/run_guidance.answer",
        "l4.decision_engine.response_payload",
        "api_query confirmation masking",
    ),
    "review": (
        "agent.quality.review/run_review via run_guidance",
        "api_query empty fallback when no review ran",
    ),
    "confidence": (
        "agent.guidance.decision/run_guidance confidence aggregation",
        "l4.decision_engine ActionRecord.confidence",
    ),
    "evidence": (
        "agent.guidance.decision/run_guidance.evidence",
        "l4.decision_engine response_payload.evidence",
    ),
    "recommended_path": (
        "agent.guidance.decision/GuidanceDecision via run_guidance",
        "api_query empty fallback when no path was produced",
    ),
}


def _evaluate_task_result_readiness(
    guidance: Any,
    task_context: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate private Contract readiness without changing CURRENT output."""
    from dy3_polaris.l5.agent_workers import (
        _AnswerCorrelation,
        _EvidenceCandidate,
        _FinalPrivateCandidateSet,
        _ReviewCandidate,
    )

    reasons: list[str] = []
    task_id = str(task_context.get("task_id") or "")
    task_state = task_state_runtime.get_task_state(task_context)
    task_events = task_state_runtime.get_task_events(task_context)
    if not task_id:
        reasons.append("missing_task_identity")
    if not task_state:
        reasons.append("missing_task_state")
    if any(
        str(event.get("task_id") or "") != task_id
        for event in task_events
    ):
        reasons.append("task_event_identity_mismatch")

    private_set = getattr(guidance, "_contract_candidate", None)
    if not isinstance(private_set, _FinalPrivateCandidateSet):
        return {
            "ready": False,
            "reasons": tuple(reasons + ["missing_agent_private_candidates"]),
        }

    evidence = private_set.evidence_candidate
    review = private_set.review_candidate
    correlation = private_set.answer_correlation

    if not isinstance(evidence, _EvidenceCandidate):
        reasons.append("missing_evidence_candidate")
    else:
        if evidence.task_id != task_id:
            reasons.append("evidence_task_identity_mismatch")
        if (
            evidence.producer
            != "agent.knowledge.generation/_run_multi_candidate_generation"
        ):
            reasons.append("evidence_producer_invalid")
        if not evidence.answer_identity:
            reasons.append("evidence_identity_missing")
        if not (evidence.context_chunks or evidence.sources):
            reasons.append("evidence_source_missing")
        if (
            evidence.stage != "selected"
            or evidence.knowledge_unavailable
            or evidence.honest_unavailable
        ):
            reasons.append("evidence_candidate_refused")

    if not isinstance(review, _ReviewCandidate):
        reasons.append("missing_review_candidate")
    else:
        if review.task_id != task_id:
            reasons.append("review_task_identity_mismatch")
        if (
            review.producer != "agent.quality.review/run_review"
            or not review.real_reviewer_executed
        ):
            reasons.append("review_producer_invalid")
        if not review.reviewed_answer_identity:
            reasons.append("review_identity_missing")
        if (
            not review.raw_status
            or not review.raw_verdict
            or not review.raw_reason
            or not review.raw_fact_check
            or not review.raw_anti_hallucination
        ):
            reasons.append("raw_review_facts_missing")
        if review.mapping_refused_reason or review.raw_status == "skipped":
            reasons.append("review_candidate_refused")

    if not isinstance(correlation, _AnswerCorrelation):
        reasons.append("missing_answer_correlation")
    else:
        if correlation.task_id != task_id:
            reasons.append("correlation_task_identity_mismatch")
        identities = (
            correlation.final_answer_identity,
            correlation.evidence_answer_identity,
            correlation.review_answer_identity,
        )
        if not all(identities) or len(set(identities)) != 1:
            reasons.append("answer_identity_mismatch")
        if not correlation.correlation or correlation.refusal_reasons:
            reasons.append("answer_correlation_refused")

    return {
        "ready": not reasons,
        "reasons": tuple(dict.fromkeys(reasons)),
    }


def _guard_query_response(
    response_data: dict[str, Any],
    task_context: dict[str, Any],
) -> dict[str, Any]:
    """校验 P0 响应与当前任务事实对齐，不构造完整 TaskResult。"""
    task_id = str(task_context.get("task_id") or "")
    task_state = task_state_runtime.get_task_state(task_context)
    if not task_id or not task_state:
        raise RuntimeError("P0 response requires a valid task identity and state")

    # Task identity/state 的唯一来源是当前 task_context，不采信 Agent 返回副本。
    response_data["task_id"] = task_id
    response_data["task_state"] = task_state

    missing = [
        field for field in _P0_RESPONSE_REQUIRED_FIELDS
        if field not in response_data
    ]
    if missing:
        raise RuntimeError(
            "P0 response missing required current fields: " + ", ".join(missing)
        )

    forbidden = [
        field for field in _P0_RESPONSE_FORBIDDEN_V2_FIELDS
        if field in response_data
    ]
    if forbidden:
        raise RuntimeError(
            "P0 response contains unimplemented V2 fields: " + ", ".join(forbidden)
        )

    task_events = response_data.get("task_events") or []
    if any(str(event.get("task_id") or "") != task_id for event in task_events):
        raise RuntimeError("P0 response task events do not match task identity")

    state_events = [
        event for event in task_events
        if event.get("event_type") == "StateChanged"
    ]
    if state_events and state_events[-1].get("state_after") != task_state:
        raise RuntimeError("P0 response task state does not match task events")
    if task_state == "COMPLETED" and not response_data.get("answer"):
        raise RuntimeError("P0 response cannot claim COMPLETED without an answer")

    release = response_data.get("quality_release")
    release = release if isinstance(release, dict) else {}
    release_status = str(release.get("status") or "DEGRADED")
    release_eligible = bool(release.get("eligible", False))
    if release_status not in {"FULL_RELEASE", "LIMITED_RELEASE"}:
        response_data["answer"] = ""
    elif not release_eligible or not response_data.get("answer"):
        raise RuntimeError("public answer requires an eligible quality release")
    if release_status == "ASK_USER":
        response_data["requires_confirmation"] = True
    elif release_status in {"REFUSE", "WITHHOLD", "DEGRADED"}:
        response_data["requires_confirmation"] = False

    _assert_public_dto(response_data)
    return response_data


# ============================================================
# 统一健康检查处理器
# ============================================================


class _UnifiedHandlers:
    """统一应用级路由处理器."""

    def __init__(
        self,
        bridge: IntegrationBridge,
        agent_runtime: Any | None = None,
        user_understanding_service: Any | None = None,
        task_store: task_state_runtime.LearningTaskStore | None = None,
    ) -> None:
        self._bridge = bridge
        self._agents = agent_runtime
        self._confirmations = ConfirmationStore()
        self._task_store = task_store or task_state_runtime.LearningTaskStore(
            Path(__file__).resolve().parents[1] / "l5" / "data" / "learning_tasks"
        )
        # Bounded, content-free runtime measurements.  Records contain only a
        # task id, operation name and elapsed milliseconds; no prompts, answer
        # text, private candidates or learner memory are recorded.
        self._runtime_measurements: list[dict[str, Any]] = []
        # 用户理解服务 (主动提问/语料提取/画像推理) — 设计文档 4.2
        from dy3_polaris.l2.user_understanding.service import UserUnderstandingService
        self._uu = user_understanding_service or UserUnderstandingService(
            profile_store={},
            profile_service=getattr(bridge, "profile_service", None),
        )

    def _record_runtime_measurement(
        self,
        *,
        task_id: str,
        operation: str,
        measurements: Mapping[str, Any],
        correlation: Mapping[str, Any] | None = None,
    ) -> None:
        allowed = {
            "practice_submit_ms",
            "answer_record_write_ms",
            "bkt_update_ms",
            "profile_update_ms",
            "teaching_memory_update_ms",
            "learner_view_build_ms",
            "learner_report_build_ms",
            "workspace_projection_build_ms",
            "query_total_ms",
            "retrieval_ms",
            "generation_ms",
            "review_ms",
        }
        values: dict[str, float] = {}
        for key, raw in measurements.items():
            if key not in allowed:
                continue
            try:
                values[key] = round(max(0.0, float(raw)), 3)
            except (TypeError, ValueError):
                continue
        if not values:
            return
        allowed_correlation = {
            "answer_record_id",
            "learner_id_hash",
            "origin_task_id",
            "resource_id",
            "attempt_purpose",
        }
        safe_correlation = {
            str(key): str(value)
            for key, value in dict(correlation or {}).items()
            if key in allowed_correlation and value not in {None, ""}
        }
        self._runtime_measurements.append({
            "task_id": str(task_id or "unscoped"),
            "operation": str(operation),
            "timestamp": time.time(),
            "measurements": values,
            "correlation": safe_correlation,
        })
        if len(self._runtime_measurements) > 500:
            del self._runtime_measurements[:-500]

    def runtime_measurement_summary(self) -> dict[str, Any]:
        """Return aggregate timings for tests/reports, never learner content."""

        grouped: dict[str, list[float]] = defaultdict(list)
        for record in self._runtime_measurements:
            for key, value in dict(record.get("measurements") or {}).items():
                grouped[str(key)].append(float(value))
        required = (
            "practice_submit_ms", "answer_record_write_ms", "bkt_update_ms",
            "profile_update_ms", "teaching_memory_update_ms",
            "learner_view_build_ms", "learner_report_build_ms",
            "workspace_projection_build_ms", "query_total_ms", "retrieval_ms",
            "generation_ms", "review_ms",
        )
        result: dict[str, Any] = {}
        for key in required:
            values = grouped.get(key, [])
            if not values:
                result[key] = {
                    "sample_count": 0,
                    "median_ms": None,
                    "p95_ms": None,
                    "max_ms": None,
                    "status": "NOT_OBSERVED",
                }
                continue
            ordered = sorted(values)
            p95_index = max(0, min(len(ordered) - 1, int((len(ordered) * 0.95) + 0.999) - 1))
            result[key] = {
                "sample_count": len(ordered),
                "median_ms": round(median(ordered), 3),
                "p95_ms": round(ordered[p95_index], 3),
                "max_ms": round(max(ordered), 3),
                "status": "OBSERVED",
            }
        return result

    def _latest_resource_task(self, learner_id: str) -> dict[str, Any] | None:
        matching = [
            task for task in self._task_store.list_tasks(learner_id, limit=50)
            if (
                self._task_release_allows_resources(task)
                and (task.get("resource_plan") or {}).get("resources")
            )
        ]
        if not matching:
            return None
        task = matching[0]
        return {
            "task_id": str(task.get("task_id") or ""),
            "resume_route": str(task.get("resume_route") or "learning"),
            "created_at": float(task.get("created_at", 0.0) or 0.0),
        }

    def _resource_plan(self, task_id: str, learner_id: str) -> dict[str, Any] | None:
        try:
            task = self._task_store.get_task(task_id, learner_id)
        except (PermissionError, ValueError):
            return None
        if not task or not self._task_release_allows_resources(task):
            return None
        plan = dict(task.get("resource_plan") or {})
        resources = plan.get("resources")
        return plan if isinstance(resources, list) else None

    @staticmethod
    def _task_release_allows_resources(task: Mapping[str, Any]) -> bool:
        reviewer = task.get("reviewer") if isinstance(task.get("reviewer"), Mapping) else {}
        release = reviewer.get("release") if isinstance(reviewer.get("release"), Mapping) else {}
        review = reviewer.get("review") if isinstance(reviewer.get("review"), Mapping) else {}
        answer = task.get("answer") if isinstance(task.get("answer"), Mapping) else {}
        return bool(
            str(release.get("status") or "") in {"FULL_RELEASE", "LIMITED_RELEASE"}
            and bool(release.get("eligible", False))
            and str(review.get("agent_id") or "") == "agent.quality.review"
            and str(review.get("status") or "") == "completed"
            and str(review.get("verdict") or "") == "approved"
            and str(answer.get("text") or "")
        )

    @classmethod
    def _public_task_record(cls, task: Mapping[str, Any]) -> dict[str, Any]:
        """Project a durable task without reviving legacy unapproved resources."""

        public = deepcopy(dict(task))
        if cls._task_release_allows_resources(public):
            return public
        public["resource_plan"] = {"resources": []}
        public["practice_plan"] = {}
        result = public.get("public_result")
        if isinstance(result, dict):
            result["learning_resources"] = []
        public["task_events"] = [
            event for event in public.get("task_events") or []
            if str((event or {}).get("event_type") or "") != "ResourceIssued"
        ]
        return public

    def _build_workspace_projection(self, learner_id: str) -> dict[str, Any]:
        """Build one authoritative workspace projection from existing layers."""

        from types import SimpleNamespace
        from dy3_polaris.l5.agent_memory import build_memory_views
        from dy3_polaris.l5.learner_intelligence import (
            build_learner_intelligence_view,
            build_public_learner_report,
        )
        from dy3_polaris.l5.teaching_memory import load_teaching_memory_view

        profile_service = getattr(self._bridge, "profile_service", None)
        irt_service = getattr(self._bridge, "irt_service", None)
        memory_views = (
            build_memory_views(profile_service, learner_id, "继续当前科研学习")
            if profile_service is not None
            else {}
        )
        l3_router = getattr(self._bridge, "l3_router", None)
        l3_store = getattr(l3_router, "_store", None)
        if l3_store is None:
            l3_store = getattr(getattr(self._bridge, "_l3_router", None), "_store", None)

        started = time.monotonic()
        learner_view = build_learner_intelligence_view(
            {
                "learner_id": learner_id,
                "query": "继续当前科研学习",
                "user_understanding_service": self._uu,
            },
            SimpleNamespace(
                profile_service=profile_service,
                irt_service=irt_service,
                memory_service=getattr(self._bridge, "memory_service", None),
                bkt_service=None,
                l3_store=l3_store,
                user_understanding_service=self._uu,
            ),
            learner_memory_view=memory_views.get("agent.learning.diagnosis"),
            teaching_memory_view=(
                load_teaching_memory_view(profile_service, learner_id)
                if profile_service is not None
                else None
            ),
        )
        learner_view_ms = (time.monotonic() - started) * 1000.0
        report_started = time.monotonic()
        report = build_public_learner_report(learner_view)
        report_ms = (time.monotonic() - report_started) * 1000.0

        l2_router = getattr(self._bridge, "_l2_router", None)
        practice_bank = getattr(getattr(l2_router, "_handlers", None), "_practice", None)
        latest = self._latest_resource_task(learner_id)
        released_concepts: set[str] = set()
        if latest is not None:
            plan = self._resource_plan(
                str(latest.get("task_id") or ""), learner_id,
            ) or {}
            for resource in plan.get("resources") or ():
                if not isinstance(resource, Mapping) or not resource.get("evidence_refs"):
                    continue
                released_concepts.update(
                    str(item) for item in resource.get("target_concepts") or () if str(item)
                )

        projection_started = time.monotonic()
        workspace = build_learning_workspace_view(
            learner_view=learner_view,
            learner_report=report,
            practice_bank=practice_bank,
            recent_task=latest,
            released_evidence_concepts=released_concepts,
        )
        projection = public_learning_workspace_projection(workspace)
        projection_ms = (time.monotonic() - projection_started) * 1000.0
        self._record_runtime_measurement(
            task_id=str((latest or {}).get("task_id") or "workspace-read"),
            operation="learning_workspace",
            measurements={
                "learner_view_build_ms": learner_view_ms,
                "learner_report_build_ms": report_ms,
                "workspace_projection_build_ms": projection_ms,
            },
        )
        _assert_public_dto(projection)
        return projection

    async def api_learning_workspace(self, request: Request) -> JSONResponse:
        learner_id = str(request.path_params.get("learner_id") or "").strip()
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少路径参数: learner_id"), status_code=400)
        try:
            return JSONResponse(_ok(self._build_workspace_projection(learner_id)))
        except Exception as exc:  # noqa: BLE001 - honest degraded public response
            _logger.exception("learning workspace projection failed")
            return JSONResponse(
                _err(-32400, "学习工作台暂不可用", "projection_failed"),
                status_code=503,
            )

    @staticmethod
    def _task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
        answer = task.get("answer") if isinstance(task.get("answer"), Mapping) else {}
        release = (
            (task.get("reviewer") or {}).get("release")
            if isinstance(task.get("reviewer"), Mapping) else {}
        )
        return {
            "task_id": str(task.get("task_id") or ""),
            "learner_id": str(task.get("learner_id") or ""),
            "brief": str(task.get("brief") or task.get("query") or "")[:160],
            "state": str(task.get("state") or ""),
            "updated_at": float(task.get("updated_at") or 0.0),
            "resume_route": str(task.get("resume_route") or "learning"),
            "answer_available": bool((answer or {}).get("text")),
            "release_status": str((release or {}).get("status") or ""),
            "next_action": dict(task.get("next_action") or {}),
        }

    async def api_learning_tasks(self, request: Request) -> JSONResponse:
        """List server-authoritative task resumptions for one learner."""

        learner_id = str(request.path_params.get("learner_id") or "").strip()
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少路径参数: learner_id"), status_code=400)
        try:
            tasks = self._task_store.list_tasks(learner_id, limit=50)
        except ValueError as exc:
            return JSONResponse(_err(-32602, "无效的 learner_id", str(exc)), status_code=400)
        return JSONResponse(_ok({
            "learner_id": learner_id,
            "tasks": [self._task_summary(task) for task in tasks],
            "total": len(tasks),
            "source": "LEARNING_TASK_STORE",
        }))

    async def api_learning_task_detail(self, request: Request) -> JSONResponse:
        """Read one durable task; learner ownership is enforced by the store."""

        learner_id = str(request.path_params.get("learner_id") or "").strip()
        task_id = str(request.path_params.get("task_id") or "").strip()
        try:
            task = self._task_store.get_task(task_id, learner_id)
        except (PermissionError, ValueError):
            task = None
        if task is None:
            return JSONResponse(_err(-32600, "任务不存在或不属于当前学习者"), status_code=404)
        return JSONResponse(_ok({
            "task": self._public_task_record(task),
            "source": "LEARNING_TASK_STORE",
        }))

    async def api_learning_task_resume(self, request: Request) -> JSONResponse:
        """Resume means reload the same task identity, never create a new task."""

        learner_id = str(request.path_params.get("learner_id") or "").strip()
        task_id = str(request.path_params.get("task_id") or "").strip()
        try:
            task = self._task_store.get_task(task_id, learner_id)
        except (PermissionError, ValueError):
            task = None
        if task is None:
            return JSONResponse(_err(-32600, "任务不存在或不属于当前学习者"), status_code=404)
        self._task_store.record_activity(
            task_id, learner_id, "TaskResumed", "api_learning_task_resume",
            {"resume_route": str(task.get("resume_route") or "learning")},
        )
        refreshed = self._task_store.get_task(task_id, learner_id)
        public_task = self._public_task_record(refreshed or {})
        return JSONResponse(_ok({
            "task": public_task,
            "resume_route": str(public_task.get("resume_route") or "learning"),
            "source": "LEARNING_TASK_STORE",
        }))

    def observe_practice_validation(
        self,
        request_body: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Correlate a scored authored question with a server-issued resource.

        Requests without a valid task/resource pair still update BKT through
        L2, but cannot claim that a teaching strategy was validated.
        """

        learner_id = str(request_body.get("learner_id") or "")
        task_id = str(request_body.get("task_id") or "")
        resource_id = str(request_body.get("resource_id") or "")
        self._record_runtime_measurement(
            task_id=task_id or "practice-unscoped",
            operation="practice_submit",
            measurements=dict(result.get("_runtime_metrics") or {}),
            correlation={
                "answer_record_id": "answer-" + hashlib.sha256(
                    f"{learner_id}|{result.get('qid') or ''}|{result.get('attempts') or 0}".encode("utf-8")
                ).hexdigest()[:20],
                "learner_id_hash": hashlib.sha256(learner_id.encode("utf-8")).hexdigest()[:20],
                "origin_task_id": task_id,
                "resource_id": resource_id,
                "attempt_purpose": str(result.get("attempt_purpose") or "DIAGNOSTIC"),
            },
        )
        plan = self._resource_plan(task_id, learner_id)
        if (
            not learner_id
            or not task_id
            or not resource_id
            or not plan
        ):
            return
        resource = next(
            (
                item for item in plan.get("resources") or ()
                if isinstance(item, Mapping)
                and str(item.get("resource_id") or "") == resource_id
            ),
            None,
        )
        if not isinstance(resource, dict):
            return
        payload = resource.get("payload") if isinstance(resource.get("payload"), dict) else {}
        target_kps = {str(item) for item in payload.get("target_kps") or () if str(item)}
        kp_id = str(result.get("kp_id") or "")
        if target_kps and kp_id not in target_kps:
            return
        question_id = str(result.get("qid") or "")
        event_id = "practice-validation-" + hashlib.sha256(
            f"{learner_id}|{task_id}|{resource_id}|{question_id}|{result.get('attempts', 0)}".encode("utf-8")
        ).hexdigest()[:20]
        event = PracticeValidationEvent(
            event_id=event_id,
            learner_id=learner_id,
            task_id=task_id,
            resource_id=resource_id,
            question_id=question_id,
            kp_id=kp_id,
            concept_ids=tuple(
                str(item) for item in resource.get("target_concepts") or () if str(item)
            ),
            strategy=f"resource:{resource.get('resource_form') or resource.get('resource_family') or 'unknown'}",
            correct=bool(result.get("correct")),
            timestamp=time.time(),
        )
        memory_started = time.monotonic()
        commit_practice_validation(
            getattr(self._bridge, "profile_service", None),
            event,
        )
        try:
            self._task_store.record_activity(
                task_id,
                learner_id,
                "PracticeSaved",
                "l2.practice",
                {
                    "resource_id": resource_id,
                    "question_id": question_id,
                    "kp_id": kp_id,
                    "correct": bool(result.get("correct")),
                    "attempt_purpose": str(result.get("attempt_purpose") or "DIAGNOSTIC"),
                    "answer_saved": bool(result.get("answer_saved", True)),
                    "model_updated": bool(result.get("model_updated", True)),
                },
            )
            task = self._task_store.get_task(task_id, learner_id) or {}
            receipts = list(task.get("practice_receipts") or [])
            receipts.append({
                "event_id": event_id,
                "question_id": question_id,
                "kp_id": kp_id,
                "correct": bool(result.get("correct")),
                "attempt_purpose": str(result.get("attempt_purpose") or "DIAGNOSTIC"),
                "answer_saved": bool(result.get("answer_saved", True)),
                "model_updated": bool(result.get("model_updated", True)),
                "timestamp": event.timestamp,
            })
            self._task_store.update_task(
                task_id, learner_id, {"practice_receipts": receipts},
            )
            if bool(result.get("model_updated", True)):
                self._task_store.record_activity(
                    task_id,
                    learner_id,
                    "LearnerProjectionUpdated",
                    "l2.practice",
                    {"kp_id": kp_id, "source": "BKT_PROFILE_PIPELINE"},
                )
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            _logger.warning("practice task receipt skipped: %s", exc)
        self._record_runtime_measurement(
            task_id=task_id,
            operation="teaching_memory_validation",
            measurements={
                "teaching_memory_update_ms": (time.monotonic() - memory_started) * 1000.0,
            },
        )

    def _get_l1_session(self, l1_session_id: str) -> Any | None:
        """获取 L1 用户会话 (统一会话入口; L2/L5 需要上下文时经此获取)."""
        try:
            l1_router = getattr(self._bridge, "l1_router", None)
            if l1_router is None:
                return None
            session_mgr = getattr(l1_router, "_session_mgr", None)
            if session_mgr is None:
                return None
            return session_mgr.get_session(l1_session_id)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("获取 L1 会话失败: %s", exc)
            return None

    def get_session_context(self, session_id: str) -> dict[str, Any] | None:
        """从 L1 统一会话获取上下文 (L2/L5 会话上下文来源).

        返回 L1 ContextEnvelope 的关键字段 (user_id/session_id/learning_state/
        mastery_snapshot 等), 供 L2/L5 内部使用.
        """
        session = self._get_l1_session(session_id)
        if session is None:
            return None
        ctx = getattr(session, "context", None)
        if ctx is None:
            return None
        to_dict = getattr(ctx, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:  # noqa: BLE001
                pass
        return vars(ctx)

    def _attach_l1_session(self, l1_session_id: str, agent_session_id: str) -> None:
        """跨层关联: 把 L5 Agent 执行会话挂到 L1 用户会话 (统一会话闭环).

        通过 L1 Router 内部接口 (进程内直调, 无 HTTP 往返), 失败不影响主链路.
        """
        try:
            l1_router = getattr(self._bridge, "l1_router", None)
            if l1_router is None:
                return
            session_mgr = getattr(l1_router, "_session_mgr", None)
            if session_mgr is None:
                return
            session = session_mgr.get_session(l1_session_id)
            if session is None:
                return
            attach = getattr(session, "attach_agent_session", None)
            if attach is not None:
                attach(agent_session_id)
            else:
                if agent_session_id not in (getattr(session, "agent_sessions", []) or []):
                    session.agent_sessions = list(getattr(session, "agent_sessions", []) or []) + [agent_session_id]
                session.question_count = int(getattr(session, "question_count", 0) or 0) + 1
        except Exception as exc:  # noqa: BLE001
            _logger.debug("L1 会话关联失败: %s", exc)

    async def unified_health(self, request: Request) -> JSONResponse:
        """GET /health — 统一健康检查 (聚合所有层).

        返回所有层的健康状态, 任一层 degraded 则整体 degraded。
        """
        layer_health = self._bridge.get_cross_layer_health()

        # 聚合状态: 任一层 degraded 则整体 degraded
        overall_status = "healthy"
        for layer_info in layer_health.values():
            if layer_info.get("status") != "healthy":
                overall_status = "degraded"
                break

        return JSONResponse(_ok({
            "status": overall_status,
            "timestamp": time.time(),
            "layers": layer_health,
        }))

    async def api_info(self, request: Request) -> JSONResponse:
        """GET /api/info — API 发现端点.

        返回所有可用 API 端点的列表, 用于客户端自动发现。
        """
        endpoints: list[dict[str, Any]] = []
        layers_found: list[str] = []

        # 按层收集端点
        layer_specs = [
            ("L0", "governance_router", "/governance"),
            ("L1", "l1_router", "/l1"),
            ("L2", "_l2_router", "/l2"),
            ("L3", "l3_router", "/l3"),
            ("L4", "_l4_router", "/l4"),
            ("L5", "_l5_router", "/l5"),
            ("L6", "l6_router", "/l6"),
            ("L7", "_l7_router", "/l7"),
        ]

        for layer_name, router_attr, prefix in layer_specs:
            router = getattr(self._bridge, router_attr, None)
            if router is not None and hasattr(router, "get_routes_summary"):
                layers_found.append(layer_name)
                for route_info in router.get_routes_summary():
                    endpoints.append({
                        "layer": layer_name,
                        "path": f"{prefix}{route_info['path']}",
                        "methods": route_info["methods"],
                        "description": route_info.get("description", ""),
                    })

        return JSONResponse(_ok({
            "endpoints": endpoints,
            "total": len(endpoints),
            "layers": layers_found,
        }))

    # ------------------------------------------------------------
    # 前端系统 (M-F2) — 根页面 / 静态资源 / 端到端查询
    # ------------------------------------------------------------

    @staticmethod
    def _static_dir() -> Path:
        """L7 静态资源目录 (l7/static)."""
        return Path(__file__).resolve().parent.parent / "l7" / "static"

    async def index_page(self, request: Request) -> FileResponse | JSONResponse:
        """GET / — 返回系统前端页面 (index.html)."""
        index = self._static_dir() / "index.html"
        if not index.exists():
            return JSONResponse(
                _err(-32410, "前端页面未构建", "l7/static/index.html 不存在"),
                status_code=503,
            )
        return FileResponse(index)

    def _collect_agent_trace(
        self,
        learner_id: str,
        started: float,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """收集本次问答的 Agent 执行轨迹 (诊断/生成/审核/决策, 就地展示).

        从 L0 AuditEngine 持久化审计中取最近 Agent 执行记录 (按时间倒序),
        使学员在答疑结果处即可看到 4 个 Agent 的协作痕迹 (无需跳转监控页).
        """
        try:
            l1_router = getattr(self._bridge, "l1_router", None)
            audit_engine = getattr(l1_router, "_audit_engine", None)
            if audit_engine is None:
                return []
            logs = audit_engine.query(limit=30)
            trace: list[dict[str, Any]] = []
            for log in logs:
                agent_id = getattr(log, "agent_id", "") or ""
                if not agent_id or not str(agent_id).startswith("agent."):
                    continue
                ts = float(getattr(log, "timestamp", 0.0) or 0.0)
                meta = getattr(log, "metadata", {}) or {}
                trace.append({
                    "agent_id": str(agent_id),
                    "result": str(getattr(log, "outcome", "") or "success"),
                    "time": ts,
                    "detail": str(meta.get("detail", "") or "")[:120],
                })
                if len(trace) >= limit:
                    break
            return trace
        except Exception as exc:  # noqa: BLE001
            _logger.debug("收集 Agent 轨迹失败: %s", exc)
            return []

    async def api_query(self, request: Request) -> JSONResponse:
        """POST /api/query — 端到端多 Agent 知识问答聚合端点.

        编排: L2 画像 → L4 决策 (意图路由→计划→执行→校验→行动选择)
              → 会话记录 → 统一响应。前端只展示结果与流水线。

        Request::
            {"query": str, "learner_id": str?, "session_id": str?, "context": {}}

        Response data::
             {task_id, task_state, answer, action_type, confidence, pipeline[], evidence[],
              session{...}?, learner{...}?, safety_level}
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)

        query = (body or {}).get("query", "")
        if not query or not str(query).strip():
            return JSONResponse(_err(-32700, "缺少必填参数: query"), status_code=400)

        learner_id = (body or {}).get("learner_id", "")
        session_id = (body or {}).get("session_id", "")
        context = (body or {}).get("context") or {}
        started = time.monotonic()
        # P0-01: 入口生成的任务身份在本次请求链内保持不变；不复用临时 plan_id。
        task_id = f"task-{uuid.uuid4().hex}"
        # Anonymous callers are request-scoped unless the client supplies a
        # stable guest/user identity.  Never share a demo learner implicitly.
        effective_learner_id = str(learner_id or f"guest-{task_id}")
        task_context = task_state_runtime.create_task_context(
            task_id,
            producer="api_query",
            learner_id=effective_learner_id,
            session_id=str(session_id or f"request-{task_id}"),
        )
        self._task_store.create_task(
            task_context=task_context,
            learner_id=effective_learner_id,
            session_id=str(session_id or f"request-{task_id}"),
            query=str(query),
            goal=str(context.get("learning_goal") or ""),
        )
        self._task_store.bind_context(task_context)

        # 0. 动态可视化意图检测 (M-F8): 用户要求画能级/跃迁/电子云图时,
        #    实时解析其话语并生成可视化数据, 随回答一并返回, 由前端动态渲染。
        # P0-02 UNDERSTANDING 的真实边界：当前请求意图/可视化判断与画像快照处理。
        task_state_runtime.set_task_state(
            task_context,
            "UNDERSTANDING",
            producer="api_query",
        )
        viz_payload: dict[str, Any] | None = None
        try:
            viz_payload = _viz_generate(str(query), body.get("viz_data"))
        except Exception as exc:  # noqa: BLE001
            _logger.debug("viz generate failed: %s", exc)

        bridge = self._bridge

        # 1. 学习者画像 (可选) — L2
        learner_snapshot: dict[str, Any] | None = None
        profile_service = getattr(bridge, "profile_service", None)
        if learner_id and profile_service is not None:
            try:
                profile = profile_service.get_profile_snapshot(learner_id)
                if profile is not None and hasattr(profile, "to_dict"):
                    learner_snapshot = profile.to_dict()
            except Exception as exc:  # noqa: BLE001
                _logger.warning("profile fetch failed for %s: %s", learner_id, exc)

        # 2. 决策链路 — 优先走导学决策 Agent，失败时回退 L4
        pipeline: list[dict[str, Any]] = [
            {"step": "画像诊断", "detail": "L2 学习者画像", "elapsed_ms": None},
        ]
        payload: dict[str, Any] = {}
        _meta: dict[str, Any] = {}
        action_record = None
        action_type = "direct_answer"
        confidence = 0.0
        plan_id = ""
        answer = ""
        recommended_path: list[dict[str, Any]] = []
        questions: list[str] = []
        reason: list[str] = []
        requires_confirmation = False
        knowledge_unavailable = False
        review: dict[str, Any] = {}
        guidance: dict[str, Any] | None = None
        quality_release: dict[str, Any] = {
            "status": "DEGRADED",
            "eligible": False,
            "message": "完整多智能体审核链暂不可用，未发布未经审核的回答。",
            "reason_codes": ["agent_runtime_unavailable"],
            "review_status": "",
            "review_verdict": "",
            "correction_count": 0,
            "evidence_versions": [],
        }
        learning_resources: list[dict[str, Any]] = []
        teaching_strategy: dict[str, Any] = {}
        agent_ok = False
        clarify: dict[str, Any] | None = None
        flow_events: list[dict[str, Any]] = []
        broadcast_events: list[dict[str, Any]] = []
        question_type: str = ""
        sources: list[dict[str, Any]] = []

        if self._agents is not None:
            try:
                # 超时兜底: 导学编排 25s 内完成, 超时回退 L4 决策引擎
                # 传递上下文记忆（对话历史 + 当前主题）
                agent_input = {
                    "query": str(query),
                    "learner_id": effective_learner_id,
                    "task_id": task_id,
                    "task_context": task_context,
                    "trace_id": get_trace_id(),
                    # 请求未携带 L1 会话时，仅为本次内部审计建立
                    # request-scoped 关联；不改变公共会话或 API 语义。
                    "session_id": str(session_id or f"request-{task_id}"),
                }
                recent_history = context.get("recent_history") or []
                topic = context.get("topic") or ""
                teaching_action = str(context.get("teaching_action") or "")
                if teaching_action:
                    agent_input["teaching_action"] = teaching_action
                if recent_history or topic or teaching_action:
                    agent_input["context"] = {
                        "recent_history": recent_history,
                        "topic": topic,
                        "teaching_action": teaching_action,
                    }
                guidance = await asyncio.wait_for(
                    self._agents.run(
                        "agent.guidance.decision",
                        agent_input,
                    ),
                    timeout=25.0,
                )
                # 流水线: 主线 + 自纠回流 (带真实耗时, 设计多线协作展示)
                clines = guidance.get("collab_lines") or []
                if clines:
                    pipeline = []
                    for cl in clines:
                        for st in cl.get("steps", []):
                            pipeline.append({
                                "step": cl.get("label", "协作线"),
                                "detail": str(st.get("agent", "")).replace("agent.", "") + " · " + str(st.get("output", "")),
                                "elapsed_ms": st.get("elapsed_ms"),
                                "line": cl.get("line", "L1"),
                            })
                else:
                    pipeline.extend([
                        {
                            "step": "意图路由",
                            "detail": "导学决策 Agent 编排：学情诊断 → 知识生成 → 审核校验",
                            "elapsed_ms": None,
                        },
                        {
                            "step": "计划执行",
                            "detail": "知识生成 Agent 检索与响应合成",
                            "elapsed_ms": None,
                        },
                        {
                            "step": "校验裁决",
                            "detail": "审核：" + str((guidance.get("review") or {}).get("verdict", "-")),
                            "elapsed_ms": None,
                        },
                    ])
                collab_lines = guidance.get("collab_lines") or []
                self_correction = guidance.get("self_correction")
                flow_events = guidance.get("flow_events") or []
                broadcast_events = guidance.get("broadcast_events") or []
                consensus_score = float(guidance.get("consensus_score", 0.0) or 0.0)
                consensus_reached = bool(guidance.get("consensus_reached", False))
                candidate_count = len(guidance.get("candidates") or [])
                candidates = guidance.get("candidates") or []
                divergence_matrix = guidance.get("divergence_matrix") or []
                debate = guidance.get("debate")
                needs_adjudication = bool(guidance.get("needs_adjudication", False))
                consensus_threshold = float(guidance.get("consensus_threshold", 0.5) or 0.5)
                question_type = guidance.get("question_type") or ""
                sources = list(guidance.get("sources") or [])
                reasoning_loop = guidance.get("reasoning_loop")
                clarify = guidance.get("clarify")
                incoming_release = guidance.get("quality_release")
                if (
                    isinstance(incoming_release, dict)
                    and incoming_release.get("status")
                ):
                    quality_release = dict(incoming_release)
                learning_resources = list(
                    guidance.get("learning_resources") or []
                )
                teaching_strategy = dict(
                    guidance.get("teaching_strategy") or {}
                )
                requires_confirmation = bool(
                    guidance.get("requires_confirmation", False)
                ) and str(quality_release.get("status") or "") == "ASK_USER"
                knowledge_unavailable = bool(guidance.get("knowledge_unavailable", False))
                # 行动类型语义统一 (策略归位 L4): 优先 L4 next-action 产出
                action_type = guidance.get("action_type") or (
                    "clarify" if clarify else (
                        "negotiate" if requires_confirmation else "direct_answer"
                    )
                )
                confidence = float(guidance.get("confidence", 0.0) or 0.0)
                plan_id = f"agent-guidance-{uuid.uuid4().hex[:12]}"
                answer = str(guidance.get("answer", "") or "")
                if (
                    str(quality_release.get("status") or "")
                    not in {"FULL_RELEASE", "LIMITED_RELEASE"}
                    or not bool(quality_release.get("eligible", False))
                ):
                    answer = ""
                review = guidance.get("review") or {}
                recommended_path = guidance.get("recommended_path") or []
                if clarify:
                    # 模糊问题 → 人性化引导式澄清 (作为对用户问题的自然补充)
                    questions = list(clarify.get("options") or [])
                elif requires_confirmation:
                    questions = [
                        str(review.get("reason") or "结果存在不确定性，请确认后继续")
                    ]
                reason = list(questions) or [
                    f"置信度不足 ({confidence:.2f})，建议向提问者确认"
                ]
                payload = {
                    "evidence": list(guidance.get("evidence") or []),
                    "_meta": {
                        "safety_level": "safe",
                        "validation_score": confidence,
                        "retry_count": 0,
                        "intent_type": "guidance",
                        "total_elapsed_ms": None,
                    },
                }
                _meta = payload["_meta"]
                agent_ok = True
            except Exception as exc:  # noqa: BLE001
                _logger.exception("导学决策 Agent 执行失败，回退 L4: %s", exc)

        if not agent_ok:
            decision_engine = getattr(bridge, "decision_engine", None)
            if decision_engine is None:
                task_state_runtime.set_task_state(task_context, "FAILED", producer="api_query")
                return JSONResponse(_err(-32400, "决策引擎未初始化"), status_code=503)
            try:
                action_record = await decision_engine.process_query(
                    query=query,
                    context_id=session_id or context.get("context_id", ""),
                    learner_profile=learner_snapshot or {},
                    query_vector=context.get("query_vector"),
                )
            except Exception as exc:  # noqa: BLE001
                _logger.exception("decision process_query failed")
                task_state_runtime.set_task_state(task_context, "FAILED", producer="api_query")
                return JSONResponse(_err(-32400, "决策处理失败", str(exc)), status_code=500)

            pipeline = [
                {"step": "画像诊断", "detail": "L2 学习者画像", "elapsed_ms": None},
            ]
            pipeline.extend([
                {
                    "step": "意图路由",
                    "detail": str(getattr(action_record, "selection_reason", "")),
                    "elapsed_ms": None,
                },
                {
                    "step": "计划执行",
                    "detail": f'plan: {getattr(action_record, "plan_id", "")}',
                    "elapsed_ms": None,
                },
                {
                    "step": "校验裁决",
                    "detail": f"validation: {getattr(action_record, 'validation_score', 0):.2f}",
                    "elapsed_ms": None,
                },
            ])
            payload = getattr(action_record, "response_payload", {}) or {}
            _meta = payload.get("_meta") or {}
            pipeline[1]["elapsed_ms"] = _meta.get("total_elapsed_ms")
            pipeline[1]["detail"] = _meta.get("intent_type", pipeline[1]["detail"])
            pipeline[2]["elapsed_ms"] = None
            pipeline[3]["detail"] = (
                f"validation: {_meta.get('validation_score', 0):.2f} · "
                f"retry: {_meta.get('retry_count', 0)}"
            )
            action_type_raw = getattr(action_record, "action_type", "")
            action_type = getattr(action_type_raw, "value", str(action_type_raw))
            confidence = float(getattr(action_record, "confidence", 0.0) or 0.0)
            plan_id = str(getattr(action_record, "plan_id", "") or "")
            answer = extract_answer(payload)
            questions = list(
                getattr(action_record, "clarification_questions", None) or []
            )
            # L4 is a compatibility fallback without the authoritative
            # Reviewer/identity chain.  It may complete internally, but it is
            # not eligible to publish a scientific answer.
            answer = ""
            action_type = "degraded"
            requires_confirmation = False
            payload["evidence"] = []
            sources = []
            recommended_path = []
            questions = []
            reason = [
                str(item) for item in (payload.get("escalation_reason") or [])
            ]
            if not reason:
                reason = list(questions)
            if not reason:
                reason = [
                    f"置信度不足 ({confidence:.2f})，建议向提问者确认"
                ]
            if task_state_runtime.get_task_state(task_context) not in {
                "PARTIAL", "FAILED", "CANCELLED",
            }:
                task_state_runtime.set_task_state(
                    task_context, "PARTIAL", producer="api_query:l4_fallback",
                )

        # 3. 会话记录 — 统一会话闭环: L1 唯一用户会话, L5 执行记录关联
        #    session_id 全部由 L1 生成并透传; 前端只持有 L1 会话 ID
        session: dict[str, Any] | None = None
        l1_session_id = session_id  # 请求携带的 session_id = L1 用户会话 ID
        session_manager = getattr(bridge, "session_manager", None)
        if session_manager is not None:
            try:
                if l1_session_id:
                    created = session_manager.create_session(
                        agent_id=context.get("agent_id", "main"),
                        learner_id=effective_learner_id,
                        source_session_id=l1_session_id,
                    )
                else:
                    created = session_manager.create_session(
                        agent_id=context.get("agent_id", "main"),
                        learner_id=effective_learner_id,
                    )
                # 跨层关联: L5 执行记录 → L1 用户会话 (L1 聚合执行记录)
                if l1_session_id:
                    self._attach_l1_session(l1_session_id, created.session_id)
                    # 会话上下文统一由 L1 提供: 返回 L1 会话信息 (前端只持有这一个 ID)
                    l1_session = self._get_l1_session(l1_session_id)
                    if l1_session is not None:
                        session = _l1_session_dict(l1_session)
                    else:
                        session = _dump(created)
                else:
                    session = _dump(created)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("session record failed: %s", exc)

        release_allows_completion = bool(
            str(quality_release.get("status") or "") in {"FULL_RELEASE", "LIMITED_RELEASE"}
            and bool(quality_release.get("eligible", False))
        )
        review_allows_completion = bool(
            str(review.get("agent_id") or "") == "agent.quality.review"
            and str(review.get("status") or "") == "completed"
            and str(review.get("verdict") or "") == "approved"
        )
        published_learning_resources = (
            learning_resources
            if (
                agent_ok
                and answer
                and release_allows_completion
                and review_allows_completion
            )
            else []
        )
        if (
            task_state_runtime.get_task_state(task_context) == "ANSWERING"
            and agent_ok
            and not requires_confirmation
            and not clarify
            and answer
            and release_allows_completion
            and review_allows_completion
        ):
            task_state_runtime.set_task_state(
                task_context,
                "COMPLETED",
                producer="api_query",
            )

        task_state = task_state_runtime.get_task_state(task_context)
        task_events = task_state_runtime.get_task_events(task_context)

        if requires_confirmation and plan_id:
            self._confirmations.put(
                PendingConfirmation(
                    plan_id=plan_id,
                    task_id=task_id,
                    task_state=task_state,
                    task_events=task_events,
                    query=str(query),
                    action_type=action_type,
                    confidence=confidence,
                    answer=answer,
                    learner_id=effective_learner_id,
                    pipeline=pipeline,
                    evidence=payload.get("evidence", []),
                    session=session,
                    learner=learner_snapshot,
                    safety_level=_meta.get("safety_level", "safe"),
                    confirmation_questions=questions,
                    reason="\n".join(reason),
                )
            )

        _readiness_result = _evaluate_task_result_readiness(
            guidance if agent_ok else None,
            task_context,
        )

        response_data: dict[str, Any] = {
            "task_id": task_id,
            "task_state": task_state,
            "task_events": task_events,
            "action_type": action_type,
            "confidence": confidence,
            "recommended_path": recommended_path,
            "safety_level": _meta.get("safety_level", "safe"),
            "pipeline": pipeline,
            "evidence": payload.get("evidence", []),
            "sources": sources,
            "session": session,
            "learner": learner_snapshot,
            "viz": viz_payload if (viz_payload and viz_payload.get("hit")) else None,
            "total_elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "requires_confirmation": requires_confirmation,
            "knowledge_unavailable": knowledge_unavailable,
            "review": review,
            "quality_release": quality_release,
            "teaching_strategy": teaching_strategy if agent_ok else {},
            "learning_resources": published_learning_resources,
            "learner_context": (
                dict(guidance.get("learner_context") or {}) if agent_ok else {}
            ),
            "knowledge_context": (
                dict(guidance.get("knowledge_context") or {}) if agent_ok else {}
            ),
            "agent_trace": (
                list(guidance.get("agent_trace") or []) if agent_ok else []
            ),
            "collab_lines": collab_lines if agent_ok else [],
            "self_correction": self_correction if agent_ok else None,
            "reasoning_loop": reasoning_loop if agent_ok else None,
            "flow_events": flow_events if agent_ok else [],
            "broadcast_events": broadcast_events if agent_ok else [],
            # Legacy collaboration visualization keys remain for compatibility,
            # but private/losing candidates and synthetic debate are never
            # projected into the learner-facing response.
            "consensus_score": consensus_score if agent_ok else 0.0,
            "consensus_reached": consensus_reached if agent_ok else False,
            "candidate_count": 0,
            "candidates": [],
            "divergence_matrix": [],
            "debate": None,
            "needs_adjudication": False,
            "consensus_threshold": consensus_threshold if agent_ok else 0.5,
            "question_type": question_type if agent_ok else "",
            "confirmation_questions": questions if requires_confirmation else [],
            "confirmation_reason": "\n".join(reason) if requires_confirmation else "",
            "plan_id": plan_id,
            "clarify": clarify if agent_ok else None,
        }
        if requires_confirmation:
            response_data["answer"] = ""
        else:
            response_data["answer"] = answer

        response_data = _guard_query_response(response_data, task_context)

        answer_identity = (
            hashlib.sha256(f"{task_id}{response_data.get('answer') or ''}".encode("utf-8")).hexdigest()
            if response_data.get("answer") else ""
        )
        scientific_grounding = dict(
            (response_data.get("knowledge_context") or {}).get("scientific_grounding") or {}
        )
        claim_evidence_source = list(scientific_grounding.get("claims") or [])
        reviewer_events = [
            event for event in task_events
            if str(event.get("event_type") or "") == "ReviewerChallengeRaised"
        ]
        revision_events = [
            event for event in task_events
            if str(event.get("event_type") or "") == "RevisionApplied"
        ]
        learning_blocks = [
            {"kind": "answer", "available": bool(response_data.get("answer"))},
            {"kind": "evidence", "count": len(response_data.get("evidence") or [])},
            {"kind": "review", "verdict": str(review.get("verdict") or "")},
            {"kind": "next_action", "count": len(recommended_path)},
        ]
        self._task_store.update_task(task_id, effective_learner_id, {
            "state": task_state,
            "task_events": task_events,
            "agent_contributions": list(response_data.get("agent_trace") or []),
            "reviewer": {
                "challenges": reviewer_events,
                "revisions": revision_events,
                "review": review,
                "release": quality_release,
            },
            "answer": {
                "identity": answer_identity,
                "version": 1 if answer_identity else 0,
                "text": str(response_data.get("answer") or ""),
            },
            "claim_evidence_source": claim_evidence_source,
            "resource_plan": {"resources": published_learning_resources},
            "practice_plan": next((
                dict(item) for item in published_learning_resources
                if isinstance(item, dict)
                and str(item.get("resource_family") or item.get("resource_form") or "")
                in {"assessment", "staged_questions", "practice"}
            ), {}),
            "teaching_decision": teaching_strategy,
            "next_action": {
                "action_type": action_type,
                "recommended_path": recommended_path,
            },
            "resume_route": "learning",
            "degradation": {} if agent_ok else {"path": "l4_fallback"},
            "learning_blocks": learning_blocks,
            "public_result": {
                key: value for key, value in response_data.items()
                if key in {
                    "task_id", "task_state", "task_events", "answer", "evidence",
                    "sources", "review", "quality_release", "confidence",
                    "action_type", "recommended_path", "learner_context",
                    "knowledge_context", "agent_trace", "collab_lines", "flow_events",
                    "self_correction", "reasoning_loop", "learning_resources",
                    "teaching_strategy", "knowledge_unavailable", "clarify",
                    "requires_confirmation", "question_type",
                }
            },
        })

        # Agent 运行审计: 登录态提问 → L1 审计日志 (监控页/对话审计可见)
        try:
            l1_gateway = getattr(bridge, "l1_gateway", None)
            if l1_gateway is not None and agent_ok:
                user = l1_gateway.authenticate(request)
                if user is not None:
                    l1_gateway.audit_agent_call(
                        user=user,
                        agent_id="agent.guidance.decision",
                        success=True,
                        detail=f"query={str(query)[:40]} answer_len={len(answer)}",
                    )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("query audit skipped: %s", exc)

        # Agent 交互记录: 把本次真实问答的协同过程写入 InteractionRecorder,
        # 供「交互总览 / 协同轨迹」展示真实使用情况 (替代前端演示数据).
        if agent_ok and self._agents is not None:
            try:
                recorder = self._agents.get_recorder()
                chain_id = recorder.start_chain(
                    session_id=str(l1_session_id or ""),
                    learner_id=effective_learner_id,
                    query=str(query),
                )
                _phase_by_agent = {
                    "agent.learning.diagnosis": "diagnosis",
                    "agent.knowledge.generation": "generation",
                    "agent.quality.review": "review",
                    "agent.guidance.decision": "decision",
                }
                _agent_names = {
                    "agent.learning.diagnosis": "学情诊断",
                    "agent.knowledge.generation": "知识生成",
                    "agent.quality.review": "审核校验",
                    "agent.guidance.decision": "导学决策",
                }

                def _agent_label(aid: str) -> str:
                    return _agent_names.get(aid) or (aid.split(".")[-1] if "." in aid else aid)

                for fe in flow_events:
                    aid = str(fe.get("agent", "") or "")
                    recorder.record_agent_execution(
                        agent_id=aid,
                        agent_name=_agent_label(aid),
                        action=str(fe.get("step") or fe.get("label") or "执行"),
                        input_data={"query": str(query)},
                        output_data={"detail": str(fe.get("output") or fe.get("label") or "")},
                        duration_ms=float(fe.get("elapsed_ms") or 0.0),
                        status="completed",
                        phase=_phase_by_agent.get(aid, "system"),
                        chain_id=chain_id,
                    )
                for be in broadcast_events:
                    pub = str(be.get("publisher", "") or "")
                    recorder.record_broadcast(
                        from_agent=pub,
                        from_name=_agent_label(pub),
                        to_agents=[str(be.get("to", ""))] if be.get("to") else [],
                        channel=str(be.get("channel", "") or ""),
                        payload_summary={"event": str(be.get("event", ""))},
                        phase=_phase_by_agent.get(pub, "system"),
                        chain_id=chain_id,
                    )
                recorder.end_chain(
                    chain_id=chain_id,
                    final_answer=str(answer or ""),
                    status="completed",
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("interaction record failed: %s", exc)

        stage_metrics: dict[str, float] = {}
        total_elapsed = float(response_data.get("total_elapsed_ms") or 0.0)
        if total_elapsed > 0.0:
            stage_metrics["query_total_ms"] = total_elapsed
        for event in flow_events:
            agent_id = str(event.get("agent") or "")
            elapsed = float(event.get("elapsed_ms") or 0.0)
            if agent_id == "agent.knowledge.generation" and elapsed > 0.0:
                stage_metrics["generation_ms"] = stage_metrics.get("generation_ms", 0.0) + elapsed
            elif agent_id == "agent.quality.review" and elapsed > 0.0:
                stage_metrics["review_ms"] = stage_metrics.get("review_ms", 0.0) + elapsed
        # CURRENT flow data does not separate retrieval from generation.  It is
        # intentionally left NOT_OBSERVED instead of copying generation time.
        self._record_runtime_measurement(
            task_id=task_id,
            operation="api_query",
            measurements=stage_metrics,
        )

        return JSONResponse(_ok(response_data))

    async def api_viz_generate(self, request: Request) -> JSONResponse:
        """POST /api/viz/generate — 动态可视化数据生成 (M-F8).

        根据用户话语 / 论文描述, 实时解析并生成能级图 / 电子云 / 跃迁概率
        可视化数据, 供前端 mf8-atomic-viz 动态渲染 (非静态预设)。

        Request::
            {"query": str, "data": {levels, transitions, orbitals}?}

        Response data::
            {hit, viz_type, data, note, parsed}
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        query = (body or {}).get("query", "")
        data = (body or {}).get("data")
        result = _viz_generate(str(query), data)
        return JSONResponse(_ok(result))

    async def api_feedback_aggregate(self, request: Request) -> JSONResponse:
        """GET /api/feedback/aggregate/{learner_id} — 跨通道反馈聚合.

        统一反馈类型 (l2.models.FeedbackType):
        - L2 画像 extras.feedback_log (Agent 隐式反馈 / 统一枚举)
        - L4 最近策略决策 (next-action 缓存)
        供前端/画像消费端按统一类型聚合分析.
        """
        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少路径参数: learner_id"), status_code=400)

        bridge = self._bridge
        feedback_log: list[dict[str, Any]] = []
        profile_service = getattr(bridge, "profile_service", None)
        if profile_service is not None:
            try:
                profile = profile_service.get_profile_snapshot(learner_id)
                if profile is not None:
                    feedback_log = list(
                        (getattr(profile, "extras", {}) or {}).get("feedback_log", []) or []
                    )
            except Exception as exc:  # noqa: BLE001
                _logger.debug("反馈聚合: 画像读取失败 %s", exc)

        # 按统一类型统计
        from collections import Counter

        by_type: dict[str, int] = dict(Counter(
            item.get("feedback_type", "unknown") for item in feedback_log
        ))

        # L4 最近策略决策 (决策反馈源)
        engine = getattr(bridge, "decision_engine", None)
        last_decision = None
        if engine is not None:
            last_decision = getattr(engine, "_last_next_action", {}).get(learner_id)

        return JSONResponse(_ok({
            "learner_id": learner_id,
            "total": len(feedback_log),
            "by_type": by_type,
            "channels": {
                "l2_feedback_log": len(feedback_log),
                "l4_next_action": bool(last_decision),
            },
            "last_decision": last_decision,
            "feedback_log": feedback_log[-50:],
        }))

    async def api_llm_config_get(self, request: Request) -> JSONResponse:
        """GET /api/llm/config — 读取当前 LLM 配置 (脱敏, 不返回明文密钥)."""
        from dy3_polaris.l3.llm_config import _runtime_summary
        return JSONResponse(_ok(_runtime_summary()))

    async def api_llm_config_set(self, request: Request) -> JSONResponse:
        """POST /api/llm/config — 设置 LLM 配置 (provider + api_key + 可选 base_url/model).

        写入运行时配置, 供后续 llm_synthesizer 调用; 返回脱敏摘要.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        provider = str(body.get("provider", "") or "").strip()
        if not provider:
            return JSONResponse(_err(-32700, "缺少必填参数: provider"), status_code=400)
        from dy3_polaris.l3.llm_config import set_runtime_config

        summary = set_runtime_config(
            provider=provider,
            api_key=str(body.get("api_key", "") or ""),
            base_url=str(body.get("base_url", "") or ""),
            model=str(body.get("model", "") or ""),
        )
        return JSONResponse(_ok(summary))

    # ---- 个性化学习资源生成 (3 种形态: 定制化讲解 / 实操指南 / 分阶测试题) ----

    async def api_learning_resource_interact(
        self,
        request: Request,
    ) -> JSONResponse:
        """Record one server-issued resource interaction as teaching evidence."""

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = str((body or {}).get("learner_id") or "")
        task_id = str((body or {}).get("task_id") or "")
        resource_id = str((body or {}).get("resource_id") or "")
        action = str((body or {}).get("action") or "")
        if not all((learner_id, task_id, resource_id, action)):
            return JSONResponse(
                _err(-32602, "缺少 learner_id/task_id/resource_id/action"),
                status_code=400,
            )
        plan = self._resource_plan(task_id, learner_id)
        if not plan:
            return JSONResponse(
                _err(-32600, "资源计划不存在、已过期或不属于当前学习者"),
                status_code=404,
            )
        resource = next((
            item for item in plan.get("resources") or []
            if isinstance(item, dict)
            and str(item.get("resource_id") or "") == resource_id
        ), None)
        if not isinstance(resource, dict):
            return JSONResponse(_err(-32600, "资源不存在"), status_code=404)
        try:
            event = build_resource_interaction_event(
                learner_id=learner_id,
                task_id=task_id,
                resource=resource,
                action=action,
            )
        except ValueError as exc:
            return JSONResponse(_err(-32602, "资源交互不可用", str(exc)), status_code=400)
        profile_service = getattr(self._bridge, "profile_service", None)
        persisted = commit_resource_interaction(profile_service, event)
        if not persisted:
            return JSONResponse(
                _err(-32400, "无法写入学习者教学记忆"),
                status_code=503,
            )
        self._task_store.record_activity(
            task_id,
            learner_id,
            "TeachingActionRequested",
            "api_learning_resource_interact",
            {
                "resource_id": resource_id,
                "action": event.action.value,
                "source_class": "OBSERVED",
            },
        )
        response: dict[str, Any] = {
            "event_id": event.event_id,
            "task_id": event.task_id,
            "resource_id": event.resource_id,
            "action": event.action.value,
            "source_class": "OBSERVED",
            "mastery_updated": False,
            "message": "已记录本次真实学习交互。",
        }
        next_teaching_action = {
            "still_confused": "still_confused",
            "change_explanation": "change_explanation",
            "request_example": "request_example",
            "deepen": "deepen",
        }.get(event.action.value)
        if next_teaching_action:
            response["next_teaching_action"] = next_teaching_action
            response["message"] = "反馈已进入 Teaching Memory；下一次解释将由 Diagnosis 重新决策。"
        elif event.action.value == "understood":
            response["verification_required"] = True
            response["message"] = "已记录自我报告；掌握度需通过真实作答验证后才会更新。"
        if event.action.value == "start_practice":
            response["practice_endpoint"] = str(
                (resource.get("payload") or {}).get("endpoint")
                or "/l2/practice/questions"
            )
            response["target_kps"] = list(
                (resource.get("payload") or {}).get("target_kps") or []
            )
        return JSONResponse(_ok(response))

    async def api_personalized_resources(self, request: Request) -> JSONResponse:
        """GET /api/personalized/resources — 个性化学习资源生成.

        Legacy compatibility projection for three resource families.

        Task-scoped personalized resources are produced by /api/query from
        Diagnosis + R06 context + the Quality Release Gate.  This endpoint
        keeps the old shape but marks templates and local authored questions
        honestly; it does not claim Reviewer approval or dynamic optima.
        1. customized_resource : 定制化讲解 (按学历深度调整)
        2. practical_guide      : 实操指南 (8 步实验流程)
        3. staged_questions     : 分阶测试题 (基础/进阶/前沿)
        """
        learner_id = request.query_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少查询参数: learner_id"), status_code=400)

        # 1. 学历背景 (L1 role)
        role = "undergrad"
        try:
            l1_router = getattr(self._bridge, "l1_router", None)
            users = getattr(l1_router, "_users", {}) if l1_router is not None else {}
            user_tuple = users.get(learner_id)
            if user_tuple is not None:
                rv = getattr(getattr(user_tuple[0], "role", None), "value", "") or ""
                if rv in ("undergrad", "graduate", "researcher", "teacher"):
                    role = rv
        except Exception:  # noqa: BLE001
            pass

        # 2. 理论测试结果 (L2 画像)
        km: dict[str, float] = {}
        weak: list[str] = []
        theta = 0.0
        level = "beginner"
        names: dict[str, str] = {}
        try:
            ps = getattr(self._bridge, "profile_service", None)
            if ps is not None:
                snap = ps.get_profile_snapshot(learner_id)
                if snap is not None:
                    km = dict(getattr(snap, "kp_mastery", {}) or {})
                    weak = list(getattr(snap, "weak_kps", []) or [])
                    theta = float(getattr(snap, "theta", 0.0) or 0.0)
                    level = str(getattr(snap, "level", "beginner") or "beginner")
                    names = dict(getattr(snap, "kp_names", {}) or {})
        except Exception:  # noqa: BLE001
            pass
        if not weak:
            weak = [k for k, v in km.items() if v < 0.6]

        _ROLE_META = {
            "undergrad": {"label": "本科生", "depth": "基础"},
            "graduate": {"label": "研究生", "depth": "进阶"},
            "researcher": {"label": "科研员", "depth": "前沿"},
            "teacher": {"label": "教师", "depth": "前沿"},
        }
        meta = _ROLE_META.get(role, _ROLE_META["undergrad"])

        return JSONResponse(_ok({
            "learner_context": {
                "learner_id": learner_id,
                "role": role,
                "role_label": meta["label"],
                "depth": meta["depth"],
                "theta": round(theta, 2),
                "level": level,
                "weak_kps": weak[:5],
            },
            "customized_resource": self._build_customized(meta, km, weak, names),
            "practical_guide": self._build_practical_guide(meta, theta),
            "staged_questions": self._build_staged_questions(learner_id, km),
        }))

    async def api_career_recommend(self, request: Request) -> JSONResponse:
        """GET /api/career/recommend?learner_id=xxx — 职业方向推荐.

        根据画像身份(role) + 能力档位(level) 匹配稀土发光材料产业链岗位,
        补齐验收硬缺口「职业方向推荐」(规划书"特色功能").
        """
        learner_id = request.query_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少查询参数: learner_id"), status_code=400)

        # 1. 身份 (L1 role)
        role = "undergrad"
        try:
            l1_router = getattr(self._bridge, "l1_router", None)
            users = getattr(l1_router, "_users", {}) if l1_router is not None else {}
            user_tuple = users.get(learner_id)
            if user_tuple is not None:
                rv = getattr(getattr(user_tuple[0], "role", None), "value", "") or ""
                if rv:
                    role = rv
        except Exception:  # noqa: BLE001
            pass

        # 2. 能力档位 (L2 level)
        level = "beginner"
        try:
            ps = getattr(self._bridge, "profile_service", None)
            if ps is not None:
                snap = ps.get_profile_snapshot(learner_id)
                if snap is not None:
                    level = str(getattr(snap, "level", "beginner") or "beginner")
        except Exception:  # noqa: BLE001
            pass

        _career_paths = [
            {"id": "rd", "title": "发光材料研发工程师", "roles": ["researcher", "graduate"], "levels": ["intermediate", "advanced"],
             "skills": ["材料合成", "发光机理", "配方优化", "光谱分析"],
             "desc": "荧光粉/发光材料新配方研发与性能优化，产业链核心研发岗。",
             "growth": ["研发助理", "研发工程师", "高级工程师", "技术专家/总监"]},
            {"id": "process", "title": "制备工艺工程师", "roles": ["undergrad", "graduate"], "levels": ["beginner", "intermediate"],
             "skills": ["高温固相法", "工艺参数", "量产放大", "良率控制"],
             "desc": "负责发光材料制备工艺的优化与量产放大。",
             "growth": ["工艺技术员", "工艺工程师", "工艺主管", "生产经理"]},
            {"id": "characterize", "title": "材料表征测试工程师", "roles": ["undergrad", "graduate"], "levels": ["beginner", "intermediate"],
             "skills": ["XRD", "PL 光谱", "SEM/TEM", "数据解读"],
             "desc": "操作 XRD/PL 等仪器，表征材料结构、光谱与形貌。",
             "growth": ["测试技术员", "表征工程师", "测试主管", "质量总监"]},
            {"id": "device", "title": "LED 器件工程师", "roles": ["researcher"], "levels": ["intermediate"],
             "skills": ["封装工艺", "色温/显色", "健康照明", "器件可靠性"],
             "desc": "面向白光 LED 封装与应用，关注健康照明与蓝光危害。",
             "growth": ["器件助理", "器件工程师", "光学主管", "应用总监"]},
            {"id": "qc", "title": "质量控制工程师", "roles": ["undergrad"], "levels": ["beginner"],
             "skills": ["检测标准", "批次一致性", "失效分析", "质量体系"],
             "desc": "把控发光材料/器件的批次质量与检测标准。",
             "growth": ["质检员", "QC 工程师", "质量主管", "质量经理"]},
            {"id": "sales", "title": "技术销售/应用工程师", "roles": ["undergrad", "graduate", "researcher"], "levels": ["intermediate"],
             "skills": ["产品选型", "客户需求", "应用方案", "沟通表达"],
             "desc": "面向 LED/显示客户，提供荧光粉选型与应用方案。",
             "growth": ["销售助理", "应用工程师", "大客户经理", "销售总监"]},
            {"id": "teacher", "title": "高校教师/教研员", "roles": ["teacher"], "levels": ["advanced"],
             "skills": ["课程教学", "实验指导", "出题命题", "学科建设"],
             "desc": "从事发光材料相关课程教学与学科建设。",
             "growth": ["助教", "讲师", "副教授", "教授"]},
            {"id": "postdoc", "title": "博士后/科研助理", "roles": ["graduate"], "levels": ["advanced"],
             "skills": ["前沿选题", "论文写作", "项目申报", "独立科研"],
             "desc": "进入课题组做前沿研究，向独立科研过渡。",
             "growth": ["博士后", "助理研究员", "副研究员", "研究员"]},
            {"id": "ip", "title": "知识产权/专利工程师", "roles": ["graduate", "researcher"], "levels": ["intermediate", "advanced"],
             "skills": ["专利撰写", "技术检索", "侵权分析", "行业调研"],
             "desc": "发光材料相关专利布局、撰写与技术调研。",
             "growth": ["专利助理", "专利工程师", "IP 主管", "IP 总监"]},
            {"id": "editor", "title": "科技编辑/科普作者", "roles": ["undergrad", "graduate", "researcher"], "levels": ["intermediate"],
             "skills": ["写作表达", "文献整理", "科普转化", "内容运营"],
             "desc": "把发光材料知识转化为科普内容或科技出版。",
             "growth": ["助理编辑", "编辑/作者", "主编", "内容总监"]},
        ]

        matches = []
        for path in _career_paths:
            role_hit = role in path["roles"]
            level_hit = level in path["levels"]
            if role_hit and level_hit:
                matches.append({
                    "id": path["id"], "title": path["title"],
                    "skills": path["skills"], "desc": path["desc"], "growth": path["growth"],
                })
        return JSONResponse(_ok({
            "learner_context": {"learner_id": learner_id, "role": role, "level": level},
            "careers": matches,
            "total": len(matches),
        }))

    async def api_experiment_guide(self, request: Request) -> JSONResponse:
        """GET /api/experiment/guide — 实验导学系统 (8 步流程 + 苏格拉底追问).

        补齐验收硬缺口「实验导学系统」(规划书"特色功能"): 引导学习者
        按 8 步完成发光材料实验, 每步用苏格拉底式追问而非直接给结论.
        """
        _guide = {
            "title": "镝（Dy）绿色健康照明发光材料合成与表征实验导学",
            "steps": [
                {"step": 1, "name": "明确实验目的",
                 "goal": "想清楚要解决什么问题（如：验证 Dy3+ 掺杂浓度对发光强度的影响）",
                 "socratic": "这个实验想回答什么问题？如果没有明确问题，实验数据会变成无意义的数字。"},
                {"step": 2, "name": "文献调研",
                 "goal": "查已有研究：常用基质（磷酸盐/硅酸盐/钒酸盐）、掺杂离子、合成方法",
                 "socratic": "别人已经做过什么？你的实验和已有研究相比，新意或差异在哪里？"},
                {"step": 3, "name": "设计实验方案",
                 "goal": "确定变量（掺杂浓度梯度）、对照组、样品数量、测量指标",
                 "socratic": "哪些变量会影响结果？你如何保证只有目标变量在变？"},
                {"step": 4, "name": "原料与仪器准备",
                 "goal": "列原料清单（基质原料、掺杂氧化物、助熔剂），确认 XRD/PL 仪器可用",
                 "socratic": "原料纯度为什么重要？称量误差会如何传递到最终结果？"},
                {"step": 5, "name": "材料合成（高温固相法）",
                 "goal": "称量→研磨混合→装坩埚→烧结（温度/时间/气氛）→冷却研磨",
                 "socratic": "为什么需要充分研磨？烧结温度和时间如何影响物相纯度？"},
                {"step": 6, "name": "表征测试（XRD + PL）",
                 "goal": "XRD 测物相纯度，PL 测激发/发射光谱、发光强度",
                 "socratic": "XRD 能告诉你什么？PL 激发和发射光谱分别测的是什么？"},
                {"step": 7, "name": "数据分析",
                 "goal": "解读发射峰（Dy3+ 蓝光 4F9/2→6H15/2、黄光 4F9/2→6H13/2），找浓度猝灭点",
                 "socratic": "发射峰强度随浓度先升后降，说明了什么？临界点在哪里？"},
                {"step": 8, "name": "结论与报告",
                 "goal": "回答最初的问题，讨论机理（浓度猝灭/能量传递），写报告",
                 "socratic": "你的数据支持还是推翻了最初假设？还有什么没解释清楚？"},
            ],
        }
        return JSONResponse(_ok(_guide))

    def _build_customized(self, meta: dict, km: dict, weak: list, names: dict) -> dict:
        """定制化讲解: 针对薄弱点, 从知识库检索并生成按学历深度的讲解."""
        from dy3_polaris.l2.kp_catalog import kp_name
        depth = meta["depth"]
        depth_intro = {
            "基础": "从基础概念讲起，配合生活化类比，先建立直观理解。",
            "进阶": "深入机理与能量传递过程，结合前沿进展展开。",
            "前沿": "聚焦机理本质与研究热点，适合教学与深度讨论。",
        }.get(depth, "")
        sections = []
        for kp in weak[:2]:
            name = names.get(kp) or kp_name(kp)
            points = []
            try:
                l3_router = getattr(self._bridge, "l3_router", None)
                retrieval = getattr(l3_router, "_retrieval", None) if l3_router is not None else None
                if retrieval is not None:
                    result = retrieval.keyword_search(name, top_k=3)
                    for item in list(getattr(result, "results", []) or [])[:2]:
                        txt = str(getattr(item, "content", None) or getattr(item, "text", "") or "")
                        if txt:
                            points.append(txt[:160])
            except Exception:  # noqa: BLE001
                pass
            if not points:
                points = []
            sections.append({
                "kp_id": kp,
                "kp_name": name,
                "mastery": round(float(km.get(kp, 0.0)), 2),
                "depth": depth,
                "key_points": points,
                "source_type": "retrieved" if points else "unknown",
                "availability": "available" if points else "evidence_unavailable",
            })
        return {
            "title": f"针对你薄弱点的定制讲解（{depth}深度）",
            "intro": depth_intro,
            "source_type": "retrieved_or_unknown",
            "provenance": ["L2:LearnerSnapshot", "L3:local_retrieval"],
            "sections": sections,
        }

    def _build_practical_guide(self, meta: dict, theta: float) -> dict:
        """科研学习检查模板：不伪造实验最优参数。"""
        depth = meta["depth"]
        steps = [
            ("明确科学问题", "定义要验证的材料、机理或性能问题", "记录适用边界", "什么数据才能回答这个问题？"),
            ("核对当地证据", "确认现有文献切片是否覆盖相同基质、掺杂条件和测试口径", "缺失条件必须显式记录", "证据是直接支持还是仅供参考？"),
            ("设计变量和对照", "列出自变量、控制变量、因变量和对照组", "不由系统伪造最佳参数", "哪个变量可能造成混淆？"),
            ("选择表征方法", "根据科学问题选择 XRD、PL 或其他已有仪器", "依照本地实验室 SOP", "每种表征能支持哪类结论？"),
            ("执行并留存原始数据", "按已批准的实验方案操作", "安全和设备参数以实验室规程为准", "原始数据是否足以重复分析？"),
            ("分析不确定性", "检查重复测量、误差、异常值和条件限制", "不用单次现象代替结论", "还有哪些替代解释？"),
            ("对照 Reviewer 关切", "把证据不足、过度外推和条件不匹配列入复核", "未解决的 critical 问题不发布", "哪个结论需要改写或拒绝？"),
            ("形成有边界的报告", "区分事实、推理、限制与后续验证", "保留证据引用和版本", "当前证据真正允许说到哪一步？"),
        ]
        detail_hint = {
            "基础": "（基础）每步已给出操作要点，先照着做，再想为什么。",
            "进阶": "（进阶）重点关注每步背后的机理，理解参数为何这样选。",
            "前沿": "（前沿）可思考如何优化工艺参数、改进材料性能。",
        }.get(depth, "")
        guide_steps = [{
            "step": i + 1,
            "name": s[0],
            "operation": s[1],
            "safety": s[2],
            "question": s[3],
        } for i, s in enumerate(steps)]
        return {
            "title": "发光材料科研学习检查模板（8 步）",
            "depth": depth,
            "hint": detail_hint,
            "source_type": "template",
            "parameter_status": "not_prescribed",
            "provenance": ["DY3:curated_research_learning_template"],
            "steps": guide_steps,
        }

    def _build_staged_questions(self, learner_id: str, km: dict) -> dict:
        """分阶测试题: 按掌握度分基础/进阶/前沿三阶出题."""
        try:
            from dy3_polaris.l2.dynamic_questions import DynamicQuestionEngine, KP_QUESTION_TEMPLATES
            from dy3_polaris.l2.kp_catalog import NODE_TO_KP
            import random as _random

            class _Bank:
                _lock = None
                by_qid: dict = {}

            eng = DynamicQuestionEngine(_Bank())
            rng = _random.Random(f"stage:{learner_id}")

            def _kp_mastery(node: str) -> float:
                return float(km.get(NODE_TO_KP.get(node, ""), 0.5))

            def _gen(stage: str, nodes: list) -> list:
                out = []
                for node in nodes[:2]:
                    q = eng.generate(node, rng)
                    if q is not None:
                        q["stage"] = stage
                        out.append(q)
                return out

            all_nodes = list(KP_QUESTION_TEMPLATES.keys())
            basic = sorted(all_nodes, key=_kp_mastery)[:4]          # 掌握度最低 → 基础
            frontier = sorted(all_nodes, key=_kp_mastery, reverse=True)[:4]  # 最高 → 前沿
            mid = [n for n in all_nodes if n not in set(basic) | set(frontier)][:4] or all_nodes[:4]

            stages = [
                {"stage": "基础", "difficulty": 1, "questions": _gen("基础", basic)},
                {"stage": "进阶", "difficulty": 2, "questions": _gen("进阶", mid)},
                {"stage": "前沿", "difficulty": 3, "questions": _gen("前沿", frontier)},
            ]
            return {
                "title": "分阶测试题（基础 → 进阶 → 前沿）",
                "source_type": "template",
                "question_source": "local_authored_question_templates",
                "completion_signal": "submitted_answer_record",
                "stages": stages,
            }
        except Exception as exc:  # noqa: BLE001
            _logger.warning("分阶测试题生成失败: %s", exc)
            return {"title": "分阶测试题", "stages": []}

    async def api_match_report(self, request: Request) -> JSONResponse:
        """GET /api/match-report/{learner_id} — 学情-资源匹配度报告 (竞赛核心可视化).

        聚合 L2 画像 + IRT 能力 + KP 目录 + ZPD 难度匹配, 输出三项竞赛要求的可视化数据:
        1. 知识盲区定位 (薄弱点 + 瓶颈 + 未覆盖知识点)
        2. 资源难度匹配曲线 (IRT 能力 θ vs 题目难度分布 + ZPD 三区)
        3. 学习路径规划图 (前置依赖 DAG + 推荐顺序)
        """
        learner_id = request.path_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少路径参数: learner_id"), status_code=400)

        bridge = self._bridge
        profile_service = getattr(bridge, "profile_service", None)
        irt_service = getattr(bridge, "irt_service", None)

        # 1. 画像快照 (kp_mastery / weak_kps / level / theta)
        profile: dict[str, Any] = {}
        if profile_service is not None:
            try:
                snap = profile_service.get_profile_snapshot(learner_id)
                if snap is not None:
                    if hasattr(snap, "to_dict"):
                        profile = snap.to_dict()
                    elif isinstance(snap, dict):
                        profile = dict(snap)
            except Exception as exc:  # noqa: BLE001
                _logger.debug("匹配报告: 画像读取失败 %s", exc)

        # 2. IRT 能力快照
        ability: dict[str, Any] = {}
        if irt_service is not None:
            try:
                ability = irt_service.get_ability_snapshot(learner_id) or {}
            except Exception as exc:  # noqa: BLE001
                _logger.debug("匹配报告: 能力读取失败 %s", exc)

        authoritative_report: dict[str, Any] = {}
        try:
            from types import SimpleNamespace
            from dy3_polaris.l5.agent_memory import build_memory_views
            from dy3_polaris.l5.learner_intelligence import (
                build_learner_intelligence_view,
                build_public_learner_report,
            )
            from dy3_polaris.l5.teaching_memory import load_teaching_memory_view

            memory_views = build_memory_views(
                profile_service,
                learner_id,
                "继续当前科研学习",
            ) if profile_service is not None else {}
            learner_view = build_learner_intelligence_view(
                {
                    "learner_id": learner_id,
                    "query": "继续当前科研学习",
                    "user_understanding_service": self._uu,
                },
                SimpleNamespace(
                    profile_service=profile_service,
                    irt_service=irt_service,
                    memory_service=getattr(bridge, "memory_service", None),
                    bkt_service=None,
                    user_understanding_service=self._uu,
                ),
                learner_memory_view=memory_views.get("agent.learning.diagnosis"),
                teaching_memory_view=(
                    load_teaching_memory_view(profile_service, learner_id)
                    if profile_service is not None
                    else None
                ),
            )
            authoritative_report = build_public_learner_report(learner_view)
        except Exception as exc:  # noqa: BLE001 - honest degraded report
            _logger.warning("权威学情投影不可用: %s", exc)

        # 3. KP 目录 (42 KP, 域/层级/名称)
        from dy3_polaris.l2.kp_catalog import (
            DOMAIN_LABELS,
            KP_DOMAIN_IDS,
            KP_LEVELS,
            KP_NAMES,
            KP_TO_DOMAIN,
            ALL_KP_IDS,
        )

        kp_mastery = dict((profile.get("kp_mastery") or {}) if profile else {})
        weak_kps = list(profile.get("weak_kps") or [])

        # 域级聚合
        domains: list[dict[str, Any]] = []
        for dom, ids in KP_DOMAIN_IDS.items():
            vals = [kp_mastery[kp] for kp in ids if kp in kp_mastery]
            avg = round(sum(vals) / len(vals), 4) if vals else None
            mastered = sum(1 for v in vals if v >= 0.8)
            learning = sum(1 for v in vals if 0.5 <= v < 0.8)
            weak = sum(1 for v in vals if 0 < v < 0.5)
            uncovered = len(ids) - len(vals)
            domains.append({
                "code": dom,
                "label": DOMAIN_LABELS.get(dom, dom),
                "kp_count": len(ids),
                "avg_mastery": avg,
                "mastered": mastered,
                "learning": learning,
                "weak": weak,
                "uncovered": uncovered,
            })

        # 知识盲区定位: 薄弱点 (按掌握度升序) + 未覆盖 (掌握度=0)
        blind_spots: list[dict[str, Any]] = []
        for kp in ALL_KP_IDS:
            has_model_evidence = kp in kp_mastery
            m = kp_mastery.get(kp)
            if not has_model_evidence or (m is not None and m < 0.5):
                blind_spots.append({
                    "kp_id": kp,
                    "name": KP_NAMES.get(kp, kp),
                    "domain": KP_TO_DOMAIN.get(kp, ""),
                    "level": KP_LEVELS.get(kp, ""),
                    "mastery": round(m, 4) if m is not None else None,
                    "type": "UNKNOWN" if not has_model_evidence else "VERIFIED_WEAKNESS_CANDIDATE",
                })
        blind_spots.sort(key=lambda x: (x["mastery"] is not None, x["mastery"] or 0.0, x["kp_id"]))

        # 资源难度匹配曲线: theta + ZPD 三区 + 知识点层级难度映射
        response_count = int(ability.get("response_count", 0) or 0)
        theta = (ability.get("theta") if response_count > 0 else None)
        se = (ability.get("se") if response_count > 0 else None)
        # 将 IRT θ ([-3,3]) 映射到 [0,1] 难度轴
        theta_norm = (
            round(max(0.0, min(1.0, (float(theta) + 3.0) / 6.0)), 4)
            if theta is not None
            else None
        )
        decision_name = str(
            (authoritative_report.get("difficulty_decision") or {}).get("decision")
            or "DIAGNOSE_FIRST"
        )
        zpd_lower = (
            round(max(0.0, theta_norm - 0.15), 4)
            if theta_norm is not None else None
        )
        zpd_upper = (
            round(min(1.0, theta_norm + 0.15), 4)
            if theta_norm is not None else None
        )
        # Difficulty inventory is counted from the actual authored PracticeBank.
        # When no IRT evidence exists, no learner marker or ZPD is invented.
        l2_router = getattr(self._bridge, "_l2_router", None)
        practice_bank = getattr(getattr(l2_router, "_handlers", None), "_practice", None)
        authored_questions = list(getattr(practice_bank, "questions", ()) or ())
        difficulty_axis = {1: 0.25, 2: 0.55, 3: 0.85}
        band_labels = {1: "基础", 2: "进阶", 3: "挑战"}
        band_counts = {1: 0, 2: 0, 3: 0}
        zone_counts = {"independent": 0, "zpd": 0, "frustration": 0, "unknown": 0}
        for question in authored_questions:
            band = int(question.get("difficulty", 1) or 1)
            band = band if band in band_counts else 2
            band_counts[band] += 1
            if theta_norm is None or zpd_lower is None or zpd_upper is None:
                zone_counts["unknown"] += 1
                continue
            question_difficulty = difficulty_axis[band]
            if question_difficulty < zpd_lower:
                zone_counts["independent"] += 1
            elif question_difficulty <= zpd_upper:
                zone_counts["zpd"] += 1
            else:
                zone_counts["frustration"] += 1
        resource_difficulty_match = {
            "learner_position": theta_norm,
            "learner_position_status": (
                "MODEL_INFERRED" if theta_norm is not None else "UNKNOWN"
            ),
            "zpd_lower": zpd_lower,
            "zpd_upper": zpd_upper,
            "authored_question_count": len(authored_questions),
            "bands": [
                {
                    "difficulty": band,
                    "axis_position": difficulty_axis[band],
                    "label": band_labels[band],
                    "question_count": band_counts[band],
                    "source_class": "AUTHORED_PRACTICE_BANK",
                }
                for band in (1, 2, 3)
            ],
            "decision": decision_name,
            "reason": str(
                (authoritative_report.get("difficulty_decision") or {}).get("reason")
                or ""
            ),
            "source_class": "MODEL_INFERRED+AUTHORED_PRACTICE_BANK",
        }
        if authoritative_report:
            authoritative_report["resource_difficulty_match"] = resource_difficulty_match

        # 学习路径规划: 前置依赖 (A 域线性, 其余域首节点为根)
        prerequisites: dict[str, list[str]] = {}
        for dom, ids in KP_DOMAIN_IDS.items():
            for i in range(1, len(ids)):
                prerequisites[ids[i]] = [ids[i - 1]]
        path: list[dict[str, Any]] = []
        for kp in ALL_KP_IDS:
            m = kp_mastery.get(kp, 0.0)
            if m >= 0.85:
                continue
            pres = prerequisites.get(kp, [])
            ready = all(kp_mastery.get(p, 0.0) >= 0.6 for p in pres)
            path.append({
                "kp_id": kp,
                "name": KP_NAMES.get(kp, kp),
                "domain": KP_TO_DOMAIN.get(kp, ""),
                "level": KP_LEVELS.get(kp, ""),
                "mastery": round(m, 4),
                "prerequisites": pres,
                "ready": ready,
            })
        path.sort(key=lambda x: (not x["ready"], x["mastery"], x["kp_id"]))

        # 总体掌握度: 从 kp_mastery 求平均 (与 /l2/profile 口径一致, 不引用不存在的字段)
        if kp_mastery:
            overall_mastery = round(sum(kp_mastery.values()) / len(kp_mastery), 4)
        else:
            overall_mastery = None

        return JSONResponse(_ok({
            "learner_id": learner_id,
            "theta": round(float(theta), 4) if theta is not None else None,
            "theta_norm": theta_norm,
            "se": round(float(se), 4) if se is not None else None,
            "level": profile.get("level", "unknown"),
            "overall_mastery": overall_mastery,
            "weak_kps": weak_kps,
            "domains": domains,
            "blind_spots": blind_spots,
            "difficulty_match": {
                "theta": theta_norm,
                "zpd_lower": zpd_lower,
                "zpd_upper": zpd_upper,
                "zpd": {"independent": zone_counts["independent"],
                        "zpd": zone_counts["zpd"],
                        "frustration": zone_counts["frustration"],
                        "unknown": zone_counts["unknown"]},
                "bands": resource_difficulty_match["bands"],
                "decision": decision_name,
                "reason": str((authoritative_report.get("difficulty_decision") or {}).get("reason") or ""),
            },
            "learning_path": path,
            "path_ready_count": sum(1 for p in path if p["ready"]),
            "report": authoritative_report,
            "source_class": "LEARNER_INTELLIGENCE_VIEW",
        }))

    async def api_query_confirm(self, request: Request) -> JSONResponse:
        """POST /api/query/confirm — 提问者对不确定结果进行确认/取消."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)

        plan_id = str((body or {}).get("plan_id", "") or "")
        if not plan_id:
            return JSONResponse(_err(-32700, "缺少必填参数: plan_id"), status_code=400)

        confirmation = self._confirmations.pop(plan_id)
        if confirmation is None:
            return JSONResponse(
                _err(-32600, "确认请求不存在或已过期"),
                status_code=404,
            )

        decision = str((body or {}).get("decision", "accept")).lower()
        if decision == "accept":
            if confirmation.learner_id:
                self._task_store.record_activity(
                    confirmation.task_id,
                    confirmation.learner_id,
                    "TaskResumed",
                    "api_query_confirm",
                    {"decision": "accept", "release_permitted": False},
                )
            return JSONResponse(_ok({
                "confirmed": True,
                "plan_id": plan_id,
                "task_id": confirmation.task_id,
                "task_state": confirmation.task_state,
                "task_events": confirmation.task_events,
                # ASK_USER confirms receipt of the clarification request; it
                # never releases the previously withheld scientific draft.
                "answer": "",
                "action_type": "clarify",
                "confidence": confirmation.confidence,
                "safety_level": confirmation.safety_level,
                "pipeline": confirmation.pipeline,
                "evidence": confirmation.evidence,
                "session": confirmation.session,
                "learner": confirmation.learner,
                "confirmation_questions": confirmation.confirmation_questions,
                "confirmation_reason": confirmation.reason,
            }))

        if confirmation.learner_id:
            try:
                self._task_store.transition_task(
                    confirmation.task_id,
                    confirmation.learner_id,
                    "CANCELLED",
                    "api_query_confirm",
                )
            except ValueError:
                pass
        return JSONResponse(_ok({
            "confirmed": False,
            "plan_id": plan_id,
            "task_id": confirmation.task_id,
            "task_state": confirmation.task_state,
            "task_events": confirmation.task_events,
            "message": "已取消输出，请补充信息后重新提问",
            "confirmation_questions": confirmation.confirmation_questions,
        }))

    # ---- 用户理解体系 API (设计文档 4/5) ----

    async def uu_extract(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/extract — 提交对话语料, 返回提取信号."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        turns = (body or {}).get("turns") or []
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            signals = self._uu.extract(learner_id, turns)
            return JSONResponse(_ok({"signals": signals}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu extract failed: %s", exc)
            return JSONResponse(_err(-32603, "提取失败"), status_code=500)

    async def uu_ask(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/ask — 主动提问."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        context = (body or {}).get("context") or {}
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            q = self._uu.ask(learner_id, context)
            return JSONResponse(_ok({"question": q}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu ask failed: %s", exc)
            return JSONResponse(_err(-32603, "提问生成失败"), status_code=500)

    async def uu_answer(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/answer — 用户回答回写."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            prof = self._uu.answer(learner_id, (body or {}).get("payload") or {})
            return JSONResponse(_ok({"profile": prof}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu answer failed: %s", exc)
            return JSONResponse(_err(-32603, "回答处理失败"), status_code=500)

    async def uu_profile(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/profile — 获取理解摘要 (软揭示)."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            ins = self._uu.insights(learner_id)
            return JSONResponse(_ok(ins))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu profile failed: %s", exc)
            return JSONResponse(_err(-32603, "画像读取失败"), status_code=500)

    async def uu_correct(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/correct — 用户纠正理解."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            prof = self._uu.correct(learner_id, (body or {}).get("payload") or {})
            return JSONResponse(_ok({"profile": prof}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu correct failed: %s", exc)
            return JSONResponse(_err(-32603, "纠正处理失败"), status_code=500)

    async def uu_clear(self, request: Request) -> JSONResponse:
        """DELETE /api/user-understanding/profile — 清除用户画像数据."""
        learner_id = request.query_params.get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            removed = self._uu.clear(learner_id)
            return JSONResponse(_ok({"removed": removed}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu clear failed: %s", exc)
            return JSONResponse(_err(-32603, "清除失败"), status_code=500)

    async def uu_guide(self, request: Request) -> JSONResponse:
        """POST /api/user-understanding/guide — 引导式咨询 (用户自己也不清楚时).

        结合 L2 学情画像 (掌握度/薄弱点) 与用户理解画像 (兴趣/目标),
        为用户生成引导式学习建议。
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_err(-32700, "请求体必须是合法 JSON"), status_code=400)
        learner_id = (body or {}).get("learner_id", "")
        if not learner_id:
            return JSONResponse(_err(-32602, "缺少必填参数: learner_id"), status_code=400)
        try:
            # 组装学情画像快照: L2 画像 + KP 名称映射
            snap: dict[str, Any] = {}
            profile_service = getattr(self._bridge, "profile_service", None)
            if profile_service is not None:
                try:
                    profile = profile_service.get_profile_snapshot(learner_id)
                    if profile is not None and hasattr(profile, "to_dict"):
                        snap = profile.to_dict()
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("uu guide profile fetch failed: %s", exc)
            try:
                from dy3_polaris.l2.kp_catalog import _KP_NAMES as _KP_NAMES_SRC
                snap["kp_names"] = dict(_KP_NAMES_SRC)
            except Exception:  # noqa: BLE001
                pass
            advice = self._uu.guide(learner_id, snap, (body or {}).get("context") or {})
            return JSONResponse(_ok({"guidance": advice}))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("uu guide failed: %s", exc)
            return JSONResponse(_err(-32603, "引导咨询失败"), status_code=500)


# ============================================================
# 演示数据播种 (M-F7 缺口补齐)
# ============================================================


def _seed_demo_learning_data(profile_service: Any) -> None:
    """播种 demo 学习者画像 (按学历背景定制: 本科生=新手低掌握, 教师=掌握高掌握).

    确定性伪随机 (sha256) 保证多次启动数据一致; 每次启动覆盖, 确保旧随机画像被更新.
    """
    from dy3_polaris.l2.models import LearnerSnapshot
    from dy3_polaris.l7.renderers._common import ALL_KP_IDS

    # 学历背景 → 画像档位 (贴合现实)
    _PERSONAS = {
        # 本科生: 新手, 能力值偏低, 掌握度偏低
        "DY20240001": {"salt": 11, "level": "beginner", "theta": -0.6, "lo": 0.15, "hi": 0.55, "style": "visual"},
        # 教师: 掌握, 能力值偏高, 掌握度偏高
        "DY20240002": {"salt": 29, "level": "advanced", "theta": 1.1, "lo": 0.55, "hi": 0.95, "style": "reading"},
        # 研究生: 略懂一点, 能力值中等
        "DY20240003": {"salt": 17, "level": "intermediate", "theta": 0.1, "lo": 0.35, "hi": 0.75, "style": "multimodal"},
        # 科研员: 熟悉, 能力值中上
        "DY20240004": {"salt": 23, "level": "advanced", "theta": 0.7, "lo": 0.45, "hi": 0.85, "style": "reading"},
    }

    def _mastery(learner_id: str, kp: str, salt: int, lo: float, hi: float) -> float:
        h = int(hashlib.sha256(f"{learner_id}:{kp}:{salt}".encode()).hexdigest(), 16)
        return round(lo + (h % 1000) / 1000.0 * (hi - lo), 3)

    for learner_id, cfg in _PERSONAS.items():
        mastery = {kp: _mastery(learner_id, kp, cfg["salt"], cfg["lo"], cfg["hi"]) for kp in ALL_KP_IDS}
        weak = [kp for kp, v in mastery.items() if v < 0.60]
        snap = LearnerSnapshot(
            learner_id=learner_id,
            snapshot_ts=time.time(),
            kp_mastery=mastery,
            theta=cfg["theta"],
            level=cfg["level"],
            learning_style=cfg["style"],
            bloom_target="understand",
            weak_kps=weak,
            confidence=0.86,
        )
        profile_service.store.save_profile(learner_id, snap)
        _logger.info("L2 demo profile seeded: %s (%d KP, level=%s, theta=%.2f)",
                     learner_id, len(mastery), cfg["level"], cfg["theta"])


def _seed_demo_knowledge(l3_store: Any, quality_manager: Any) -> None:
    """播种 3 个稀土发光材料实体 + 溯源链 (integrity_hash).

    幂等: 实体已存在则跳过。
    """
    from dy3_polaris.l3.models import EntityType, KnowledgeEntity

    demos = [
        {
            "name": "NaGdF4:Eu3+ 纳米晶",
            "entity_type": EntityType.MATERIAL,
            "domain": "A",
            "description": "氟化物基质上转换发光纳米材料，用于生物成像",
            "chain": [
                ("synthesize", "Agent-Synth", "共沉淀法合成, 650℃ 焙烧"),
                ("characterize", "Agent-Lab", "XRD/SEM 表征, 结晶度 92%"),
                ("verify", "Agent-QA", "质检通过: 量子效率 0.74"),
            ],
        },
        {
            "name": "YPO4:Dy3+ 荧光粉",
            "entity_type": EntityType.MATERIAL,
            "domain": "B",
            "description": "磷酸盐基质 Dy3+ 掺杂荧光粉，白光 LED 用",
            "chain": [
                ("synthesize", "Agent-Synth", "高温固相法, 1200℃ 烧结"),
                ("characterize", "Agent-Lab", "发射光谱 574nm 主峰"),
            ],
        },
        {
            "name": "BaMgAl10O17:Eu2+ (BAM)",
            "entity_type": EntityType.MATERIAL,
            "domain": "B",
            "description": "铝酸盐基质蓝色荧光粉，PDP 与 LED 背光应用",
            "chain": [
                ("synthesize", "Agent-Synth", "固相法合成, 还原气氛 1300℃"),
                ("verify", "Agent-QA", "色坐标 (0.146, 0.054)"),
            ],
        },
    ]

    seeded = 0
    chain_prev: list[str] = []  # 前序实体 ID (derived_from 串联溯源链)
    for demo in demos:
        existing = l3_store.entity_store.find_by_name(demo["name"])
        if existing:
            entity_id = existing[0].entity_id
            chain_prev.append(entity_id)
            continue
        entity = KnowledgeEntity(
            name=demo["name"],
            entity_type=demo["entity_type"],
            domain=demo["domain"],
            description=demo["description"],
        )
        created = l3_store.add_entity(entity, check_duplicate=False)
        # 单条溯源记录 (同一实体多次 record 会覆盖, 故合并为一条),
        # 经 derived_from 串联形成多跳链: 实体N → 实体N-1 → ... → 实体1
        activities = "; ".join(f"{a}@{ag}: {d}" for a, ag, d in demo["chain"])
        try:
            quality_manager.record_provenance(
                entity_id=created.entity_id,
                activity_type=demo["chain"][-1][0],
                agent_id=demo["chain"][-1][1],
                description=activities,
                derived_from=list(chain_prev),
            )
        except Exception:
            _logger.exception("seed provenance failed for %s", created.entity_id)
        chain_prev.append(created.entity_id)
        seeded += 1
        _logger.info("L3 demo entity + provenance chain seeded: %s (%s)", created.entity_id, demo["name"])
    if seeded:
        _logger.info("L3 demo knowledge seeded: %d entities", seeded)


def _seed_domain_hierarchy(l3_store: Any) -> int:
    """播种稀土发光材料 L1-L4 层级知识图谱实体与关系.

    层级: L1 发光材料 → L2 材料大类 → L3 基质体系 → L4 具体材料 (用 domain 编码层级).
    横向: 激活剂离子 (doped_with) / 发光特性 (has_property) / 应用 (used_in).
    幂等: 按名称去重, 三元组重复时跳过.
    """
    from dy3_polaris.l3.models import (
        EntityType,
        KnowledgeEntity,
        KnowledgeTriple,
    )

    E = EntityType

    # 先删后播: 清掉旧的层级实体 (domain 命中层级前缀), 避免精简后残留旧节点
    _stale_ids = {
        e.entity_id
        for e in list(l3_store.entity_store._entities.values())
        if (e.domain or "") in ("L1", "activator", "property", "application")
        or (e.domain or "").startswith(("L2:", "L3:", "L4:"))
    }
    if _stale_ids:
        for _tid, _t in list(l3_store.triple_store._triples.items()):
            if _t.subject_id in _stale_ids or _t.object_id in _stale_ids:
                try:
                    l3_store.remove_triple(_tid)
                except Exception:
                    pass
        for _eid in _stale_ids:
            try:
                l3_store.entity_store.remove_entity(_eid)
            except Exception:
                pass
        _logger.info("清理旧层级实体: %d 个", len(_stale_ids))

    # (name, entity_type, domain, description)
    entities = [
        ("发光材料", E.CONCEPT, "L1", "绿色健康照明与显示用发光材料总域"),
        # L2 材料大类 (横向可扩展, 精选代表)
        ("氧化物基质", E.CONCEPT, "L2:oxide", "稀土氧化物基质体系"),
        ("氟化物基质", E.CONCEPT, "L2:fluoride", "氟化物上转换基质体系"),
        ("磷酸盐基质", E.CONCEPT, "L2:phosphate", "磷酸盐白光基质体系"),
        ("铝酸盐基质", E.CONCEPT, "L2:aluminate", "铝酸盐蓝光基质"),
        # L3 基质体系 (每大类 1 个代表)
        ("Y₂O₃", E.CONCEPT, "L3:oxide", "氧化钇基质"),
        ("NaYF₄", E.CONCEPT, "L3:fluoride", "氟化钇钠基质"),
        ("Ca₇NaY(PO₄)₆", E.CONCEPT, "L3:phosphate", "CNYP 磷酸盐基质"),
        ("BaMgAl₁₀O₁₇", E.CONCEPT, "L3:aluminate", "BAM 铝酸盐基质"),
        # L4 具体材料 (典型代表)
        ("Y₂O₃:Eu³⁺", E.MATERIAL, "L4:oxide", "红光荧光粉"),
        ("NaYF₄:Yb³⁺,Er³⁺", E.MATERIAL, "L4:fluoride", "上转换纳米晶"),
        ("CNYP:0.07Dy³⁺", E.MATERIAL, "L4:phosphate", "白光荧光粉"),
        ("BAM:Eu²⁺", E.MATERIAL, "L4:aluminate", "蓝光荧光粉"),
        # 激活剂离子 (典型)
        ("Dy³⁺", E.CHEMICAL_COMPOUND, "activator", "白光激活剂"),
        ("Eu³⁺", E.CHEMICAL_COMPOUND, "activator", "红光激活剂"),
        ("Eu²⁺", E.CHEMICAL_COMPOUND, "activator", "宽带蓝光激活剂"),
        ("Yb³⁺", E.CHEMICAL_COMPOUND, "activator", "上转换敏化剂"),
        ("Er³⁺", E.CHEMICAL_COMPOUND, "activator", "上转换发光中心"),
        # 离子发光特性
        ("白光发射", E.CONCEPT, "property", "多带发射叠加成白光"),
        ("蓝光发射", E.CONCEPT, "property", "约 480nm 磁偶极跃迁"),
        ("红光发射", E.CONCEPT, "property", "约 659nm 红光跃迁"),
        ("上转换发光", E.CONCEPT, "property", "反斯托克斯发光"),
        # 应用场景
        ("白光LED", E.CONCEPT, "application", "通用照明"),
        ("显示背光", E.CONCEPT, "application", "显示器件背光"),
        ("生物成像", E.CONCEPT, "application", "上转换荧光成像"),
    ]

    ids: dict[str, str] = {}
    seeded = 0
    for name, etype, domain, desc in entities:
        existing = l3_store.entity_store.find_by_name(name)
        if existing:
            ids[name] = existing[0].entity_id
            continue
        ent = KnowledgeEntity(name=name, entity_type=etype, domain=domain, description=desc)
        created = l3_store.add_entity(ent, check_duplicate=False)
        ids[name] = created.entity_id
        seeded += 1

    # (subject, predicate, object) — 纵向 part_of + 横向 doped_with / has_property / used_in
    relations = [
        ("氧化物基质", "part_of", "发光材料"),
        ("氟化物基质", "part_of", "发光材料"),
        ("磷酸盐基质", "part_of", "发光材料"),
        ("铝酸盐基质", "part_of", "发光材料"),
        ("Y₂O₃", "part_of", "氧化物基质"),
        ("NaYF₄", "part_of", "氟化物基质"),
        ("Ca₇NaY(PO₄)₆", "part_of", "磷酸盐基质"),
        ("BaMgAl₁₀O₁₇", "part_of", "铝酸盐基质"),
        ("Y₂O₃:Eu³⁺", "part_of", "Y₂O₃"),
        ("NaYF₄:Yb³⁺,Er³⁺", "part_of", "NaYF₄"),
        ("CNYP:0.07Dy³⁺", "part_of", "Ca₇NaY(PO₄)₆"),
        ("BAM:Eu²⁺", "part_of", "BaMgAl₁₀O₁₇"),
        ("Y₂O₃:Eu³⁺", "doped_with", "Eu³⁺"),
        ("NaYF₄:Yb³⁺,Er³⁺", "doped_with", "Yb³⁺"),
        ("NaYF₄:Yb³⁺,Er³⁺", "doped_with", "Er³⁺"),
        ("CNYP:0.07Dy³⁺", "doped_with", "Dy³⁺"),
        ("BAM:Eu²⁺", "doped_with", "Eu²⁺"),
        ("Y₂O₃:Eu³⁺", "has_property", "红光发射"),
        ("NaYF₄:Yb³⁺,Er³⁺", "has_property", "上转换发光"),
        ("CNYP:0.07Dy³⁺", "has_property", "白光发射"),
        ("BAM:Eu²⁺", "has_property", "蓝光发射"),
        ("CNYP:0.07Dy³⁺", "used_in", "白光LED"),
        ("BAM:Eu²⁺", "used_in", "显示背光"),
        ("NaYF₄:Yb³⁺,Er³⁺", "used_in", "生物成像"),
        ("Y₂O₃:Eu³⁺", "used_in", "白光LED"),
    ]

    added_triples = 0
    for s, p, o in relations:
        sid, oid = ids.get(s), ids.get(o)
        if not sid or not oid:
            continue
        try:
            l3_store.add_triple(KnowledgeTriple(subject_id=sid, predicate=p, object_id=oid))
            added_triples += 1
        except Exception:
            pass

    if seeded or added_triples:
        _logger.info("L3 层级知识播种: 新增 %d 实体, %d 关系", seeded, added_triples)
    return seeded


# ============================================================
# UnifiedApp
# ============================================================


class UnifiedApp:
    """统一应用组装器 — 将所有层 Router 组装为单一 Starlette 应用.

    融合世界先进方案的统一应用架构:
    - Knewton SOA: 所有服务通过单一入口暴露, 前缀路由隔离
    - Duolingo EDA: 统一健康检查 + 事件总线集成
    - LangGraph: 图节点组装 + 统一状态管理

    将 L2/L4/L5 各层 Router 通过 Mount 组装到单一 Starlette 应用,
    并提供统一健康检查和 API 发现端点。

    Args:
        irt_service: L2 IRT 能力评估服务.
        profile_service: L2 画像构建服务 (可选).
        memory_service: L2 记忆管理服务 (可选).
        bkt_service: L2 BKT 知识追踪服务 (可选).
        decision_engine: L4 决策引擎.
        orchestration_engine: L5 编排引擎 (可选).
        session_manager: L5 会话管理器 (可选).
        message_bus: L5 消息总线 (可选).
        agent_runtime: L5 默认 Agent 运行时 (可选).
        l1_gateway: L1 Agent 安全网关 (可选).
        cors_origins: CORS 允许的源 (默认 ["*"]).
    """

    def __init__(
        self,
        irt_service: Any | None = None,
        profile_service: Any | None = None,
        memory_service: Any | None = None,
        bkt_service: Any | None = None,
        decision_engine: Any | None = None,
        orchestration_engine: Any | None = None,
        session_manager: Any | None = None,
        message_bus: Any | None = None,
        agent_runtime: Any | None = None,
        l1_gateway: Any | None = None,
        cors_origins: list[str] | None = None,
        governance_router: Any | None = None,
        l1_router: Any | None = None,
        l3_router: Any | None = None,
        l6_router: Any | None = None,
        l7_router: Any | None = None,
        user_understanding_service: Any | None = None,
        learning_task_store: task_state_runtime.LearningTaskStore | None = None,
    ) -> None:
        """初始化统一应用组装器.

        Args:
            irt_service: L2 IRT 能力评估服务.
            profile_service: L2 画像构建服务 (可选).
            memory_service: L2 记忆管理服务 (可选).
            bkt_service: L2 BKT 知识追踪服务 (可选).
            decision_engine: L4 决策引擎.
            orchestration_engine: L5 编排引擎 (可选).
            session_manager: L5 会话管理器 (可选).
            message_bus: L5 消息总线 (可选).
            agent_runtime: L5 默认 Agent 运行时 (可选).
            l1_gateway: L1 Agent 安全网关 (可选).
            cors_origins: CORS 允许的源 (默认 ["*"]).
            governance_router: L0 治理路由器 (可选).
            l1_router: L1 用户域路由器 (可选).
            l3_router: L3 知识层路由器 (可选).
            l6_router: L6 协议基础设施路由器 (可选).
        """
        self._cors_origins = cors_origins or ["*"]
        self._agent_runtime = agent_runtime
        self._l1_gateway = l1_gateway

        # 创建各层 Router
        self._l2_router: L2Router | None = None
        if irt_service is not None:
            self._l2_router = L2Router(
                irt_service=irt_service,
                bkt_service=bkt_service,
                profile_service=profile_service,
                memory_service=memory_service,
                cors_origins=self._cors_origins,
            )

        self._l4_router: L4Router | None = None
        if decision_engine is not None:
            self._l4_router = L4Router(
                decision_engine=decision_engine,
                cors_origins=self._cors_origins,
                profile_service=profile_service,
            )

        self._l5_router: L5Router | None = None
        from dy3_polaris.l5.agent_workers import AgentDependencies as _AgentDeps
        from dy3_polaris.l5.skill_executor import SkillExecutor

        # 技能执行器先以空依赖创建 (create_full_app_builder 组装后注入真实依赖)
        skill_executor = SkillExecutor(_AgentDeps())
        self._l5_router = L5Router(
            orchestration_engine=orchestration_engine,
            session_manager=session_manager,
            message_bus=message_bus,
            agent_runtime=agent_runtime,
            l1_gateway=l1_gateway,
            skill_executor=skill_executor,
            cors_origins=self._cors_origins,
        )
        self._skill_executor = skill_executor

        # 扩展层 Router (L0/L1/L3/L6/L7)
        self._governance_router = governance_router
        self._l1_router = l1_router
        self._l3_router = l3_router
        self._l6_router = l6_router
        self._l7_router = l7_router
        if user_understanding_service is None:
            from dy3_polaris.l2.user_understanding.service import UserUnderstandingService

            user_understanding_service = UserUnderstandingService(
                profile_store={},
                profile_service=profile_service,
            )
        self._user_understanding_service = user_understanding_service

        # 创建集成桥接器
        self._bridge = IntegrationBridge(
            irt_service=irt_service,
            profile_service=profile_service,
            memory_service=memory_service,
            decision_engine=decision_engine,
            orchestration_engine=orchestration_engine,
            session_manager=session_manager,
            message_bus=message_bus,
            governance_router=governance_router,
            l1_router=l1_router,
            l3_router=l3_router,
            l6_router=l6_router,
        )

        # 将 Router 引用附加到 bridge (用于 API 发现)
        self._bridge._l2_router = self._l2_router  # type: ignore[attr-defined]
        self._bridge._l4_router = self._l4_router  # type: ignore[attr-defined]
        self._bridge._l5_router = self._l5_router  # type: ignore[attr-defined]
        self._bridge._l7_router = self._l7_router  # type: ignore[attr-defined]

        # 统一处理器
        self._handlers = _UnifiedHandlers(
            bridge=self._bridge,
            agent_runtime=self._agent_runtime,
            user_understanding_service=self._user_understanding_service,
            task_store=learning_task_store,
        )
        if self._l2_router is not None:
            self._l2_router._handlers._practice_observer = (  # type: ignore[attr-defined]
                self._handlers.observe_practice_validation
            )

    @property
    def bridge(self) -> IntegrationBridge:
        """获取集成桥接器实例."""
        return self._bridge

    def create_app(self) -> Starlette:
        """创建统一的 Starlette 应用实例.

        将所有层 Router 通过 Mount 组装, 并添加统一健康检查和 API 发现端点。

        Returns:
            配置好的 Starlette 应用, 可直接传给 uvicorn.run()。
        """
        routes: list[Mount | Route] = []

        # 统一端点
        routes.append(
            Route("/health", self._handlers.unified_health, methods=["GET"])
        )
        routes.append(
            Route("/api/info", self._handlers.api_info, methods=["GET"])
        )
        # 端到端多 Agent 查询 (M-F2, 前端知识问答)
        routes.append(
            Route("/api/query", self._handlers.api_query, methods=["POST"])
        )
        routes.append(
            Route(
                "/api/learning-workspace/{learner_id}",
                self._handlers.api_learning_workspace,
                methods=["GET"],
            )
        )
        routes.append(
            Route(
                "/api/learning-tasks/{learner_id}",
                self._handlers.api_learning_tasks,
                methods=["GET"],
            )
        )
        routes.append(
            Route(
                "/api/learning-tasks/{learner_id}/{task_id}",
                self._handlers.api_learning_task_detail,
                methods=["GET"],
            )
        )
        routes.append(
            Route(
                "/api/learning-tasks/{learner_id}/{task_id}/resume",
                self._handlers.api_learning_task_resume,
                methods=["POST"],
            )
        )
        routes.append(
            Route(
                "/api/query/confirm",
                self._handlers.api_query_confirm,
                methods=["POST"],
            )
        )
        # LLM 配置 (前端 API 配置页: 读取/设置运行时 key, 脱敏)
        routes.append(
            Route("/api/llm/config", self._handlers.api_llm_config_get, methods=["GET"])
        )
        routes.append(
            Route("/api/llm/config", self._handlers.api_llm_config_set, methods=["POST"])
        )
        # 个性化学习资源生成 (3 种形态: 定制化讲解/实操指南/分阶测试题)
        routes.append(
            Route("/api/personalized/resources", self._handlers.api_personalized_resources, methods=["GET"])
        )
        routes.append(
            Route(
                "/api/learning/resources/interact",
                self._handlers.api_learning_resource_interact,
                methods=["POST"],
            )
        )
        # 职业方向推荐 (验收硬缺口: 按画像身份+能力匹配产业链岗位)
        routes.append(
            Route("/api/career/recommend", self._handlers.api_career_recommend, methods=["GET"])
        )
        # 实验导学系统 (验收硬缺口: 8 步流程 + 苏格拉底追问)
        routes.append(
            Route("/api/experiment/guide", self._handlers.api_experiment_guide, methods=["GET"])
        )
        # 动态可视化数据生成 (M-F8): 用户指令/论文描述 → 能级/跃迁/电子云
        routes.append(
            Route(
                "/api/viz/generate",
                self._handlers.api_viz_generate,
                methods=["POST"],
            )
        )
        # 跨通道反馈聚合 (统一反馈类型单点: l2.models.FeedbackType)
        routes.append(
            Route(
                "/api/feedback/aggregate/{learner_id}",
                self._handlers.api_feedback_aggregate,
                methods=["GET"],
            )
        )
        # 学情-资源匹配度报告 (竞赛核心可视化: 盲区定位 + 难度匹配曲线 + 路径规划图)
        routes.append(
            Route(
                "/api/match-report/{learner_id}",
                self._handlers.api_match_report,
                methods=["GET"],
            )
        )
        # 用户理解体系 (主动提问/语料提取/画像/纠正/清除) — 设计文档 5
        routes.append(
            Route("/api/user-understanding/extract", self._handlers.uu_extract, methods=["POST"])
        )
        routes.append(
            Route("/api/user-understanding/ask", self._handlers.uu_ask, methods=["POST"])
        )
        routes.append(
            Route("/api/user-understanding/answer", self._handlers.uu_answer, methods=["POST"])
        )
        routes.append(
            Route("/api/user-understanding/profile", self._handlers.uu_profile, methods=["POST"])
        )
        routes.append(
            Route("/api/user-understanding/correct", self._handlers.uu_correct, methods=["POST"])
        )
        routes.append(
            Route("/api/user-understanding/profile", self._handlers.uu_clear, methods=["DELETE"])
        )
        routes.append(
            Route("/api/user-understanding/guide", self._handlers.uu_guide, methods=["POST"])
        )

        # 前端系统 (M-F2): 根路径返回控制台页面 + 静态资源
        routes.append(
            Route("/", self._handlers.index_page, methods=["GET"])
        )
        static_dir = self._handlers._static_dir()
        if static_dir.is_dir():
            routes.append(
                Mount("/static", app=StaticFiles(directory=str(static_dir)), name="static")
            )
        elif static_dir.exists() is False:
            static_dir.mkdir(parents=True, exist_ok=True)

        # L0 治理层路由
        if self._governance_router is not None:
            routes.append(
                Mount("/governance", app=self._governance_router.create_app())
            )

        # WebSocket 三通道 (M-F5): /ws/stream, /ws/broadcast, /ws/debate
        from dy3_polaris.l7.api.websocket import HUB, websocket_routes

        # 复用 L1 JWTManager 验证握手 token (与登录态一致)
        jwt = getattr(self._l1_router, "_jwt", None)
        if jwt is None:
            jwt = getattr(self._l1_router, "jwt_manager", None)
        if jwt is not None:
            HUB.token_manager = jwt
        routes.extend(websocket_routes())

        # L1 用户域路由
        if self._l1_router is not None:
            routes.append(
                Mount("/l1", app=self._l1_router.create_app())
            )

        # L2 层路由
        if self._l2_router is not None:
            routes.append(
                Mount("/l2", app=self._l2_router.create_app())
            )

        # L3 知识层路由
        if self._l3_router is not None:
            routes.append(
                Mount("/l3", app=self._l3_router.create_app())
            )

        # L4 层路由
        if self._l4_router is not None:
            routes.append(
                Mount("/l4", app=self._l4_router.create_app())
            )

        # L5 层路由
        if self._l5_router is not None:
            routes.append(
                Mount("/l5", app=self._l5_router.create_app())
            )

        # L6 协议基础设施路由
        if self._l6_router is not None:
            routes.append(
                Mount("/l6", app=self._l6_router.create_app())
            )

        # L7 体验呈现层路由 (M-F2 挂载)
        if self._l7_router is not None:
            routes.append(
                Mount("/l7", app=self._l7_router.create_app())
            )

        # 中间件: TraceID (内层) + 幂等键 + 安全网关 + CORS (最外层, 预检放行)
        middleware = []
        middleware.append(
            Middleware(
                TraceIDMiddleware,
            )
        )
        # 幂等键: X-Idempotency-Key 同键同路径去重 (跨层写用例)
        middleware.append(
            Middleware(
                IdempotencyMiddleware,
            )
        )
        # 安全网关: 复用 L1 JWTManager 校验写端点 (白名单放行学生操作/只读检索)
        l1_jwt = getattr(self._l1_router, "_jwt", None)
        if l1_jwt is None:
            l1_jwt = getattr(self._l1_router, "jwt_manager", None)
        middleware.append(
            Middleware(
                SecurityGatewayMiddleware,
                jwt_manager=l1_jwt,
            )
        )
        if self._cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._cors_origins,
                    allow_methods=["*"] if "*" in self._cors_origins
                                   else ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)

        # 500 兜底: 未捕获异常 → 结构化错误 + trace_id 回填
        async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
            _logger.exception("未处理异常: %s", exc)
            return JSONResponse(
                _err(-32603, "Internal Server Error",
                     str(exc), trace_id=get_trace_id()),
                status_code=500,
            )

        app.add_exception_handler(Exception, _unhandled_exception)
        return app

    def get_routes_summary(self) -> list[dict[str, Any]]:
        """获取所有路由摘要 (用于文档/发现).

        Returns:
            所有层的路由列表, 每项含 layer/path/methods/description。
        """
        endpoints: list[dict[str, Any]] = []

        # 统一端点
        endpoints.append({
            "layer": "Unified",
            "path": "/health",
            "methods": ["GET"],
            "description": "统一健康检查 (聚合所有层)",
        })
        endpoints.append({
            "layer": "Unified",
            "path": "/api/info",
            "methods": ["GET"],
            "description": "API 发现端点",
        })
        endpoints.append({
            "layer": "Unified",
            "path": "/api/query",
            "methods": ["POST"],
            "description": "端到端多 Agent 知识问答聚合端点",
        })
        endpoints.append({
            "layer": "Unified",
            "path": "/api/query/confirm",
            "methods": ["POST"],
            "description": "不确定结果向提问者确认/取消",
        })
        endpoints.append({
            "layer": "Unified",
            "path": "/",
            "methods": ["GET"],
            "description": "系统前端控制台页面",
        })

        # 各层端点
        layer_specs = [
            ("L0", self._governance_router, "/governance"),
            ("L1", self._l1_router, "/l1"),
            ("L2", self._l2_router, "/l2"),
            ("L3", self._l3_router, "/l3"),
            ("L4", self._l4_router, "/l4"),
            ("L5", self._l5_router, "/l5"),
            ("L6", self._l6_router, "/l6"),
            ("L7", self._l7_router, "/l7"),
        ]

        for layer_name, router, prefix in layer_specs:
            if router is not None and hasattr(router, "get_routes_summary"):
                for route_info in router.get_routes_summary():
                    endpoints.append({
                        "layer": layer_name,
                        "path": f"{prefix}{route_info['path']}",
                        "methods": route_info["methods"],
                        "description": route_info.get("description", ""),
                    })

        return endpoints

    # ============================================================
    # 全栈工厂方法
    # ============================================================

    @classmethod
    def create_full_app_builder(cls, data_dir: str | None = None) -> UnifiedApp:
        """创建挂载全部七层的 UnifiedApp 实例.

        使用各层默认配置自动初始化并组装 L0→L6 全部路由器,
        返回可直接调用 ``create_app()`` 的 UnifiedApp 实例。

        Args:
            data_dir: 数据持久化根目录 (默认: 项目源码 data 目录).
                测试可传入临时目录以隔离生产数据 (根治测试污染).

        Returns:
            配置好全部七层 Router 的 UnifiedApp 实例。
        """
        # L0 — 治理路由器
        from dy3_polaris.l0.governance_router import (
            GovernanceRouter,
            GovernanceSubsystems,
        )
        from dy3_polaris.l0.governance import (
            PolicyStore,
            PolicyEvaluator,
            AuditEngine,
            MetricsEngine,
            ComplianceReporter,
        )
        from dy3_polaris.l0.cc1.pipeline import AntiHallucinationPipeline
        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline
        from dy3_polaris.l0.cc2.engine import CollaborationEngine

        l0_subsys = GovernanceSubsystems(
            policy_store=PolicyStore(),
            policy_evaluator=PolicyEvaluator(PolicyStore()),
            anti_hallucination_pipeline=AntiHallucinationPipeline(),
            review_pipeline=ReviewPipeline(),
            collaboration_engine=CollaborationEngine(),
            audit_engine=AuditEngine(
                persist_path=str(
                    (Path(data_dir) if data_dir else Path(__file__).resolve().parents[1])
                    / "l0" / "data" / "audit.jsonl"
                )
            ),
            metrics_engine=MetricsEngine(),
            compliance_reporter=ComplianceReporter(),
        )
        governance_router = GovernanceRouter(l0_subsys)

        # L1 — 用户域路由器
        from dy3_polaris.l1.api_integration import L1APIRouter
        from dy3_polaris.l1.auth import JWTManager, PasswordHasher
        from dy3_polaris.l1.access_control import AccessControlManager
        from dy3_polaris.l1.models import User, UserRole, ABACAttributes

        l1_router = L1APIRouter(
            jwt_manager=JWTManager(),
            access_control=AccessControlManager(),
            audit_engine=l0_subsys.audit_engine,
        )
        # 注册演示用户 (M-F2 种子数据)
        l1_router.register_user(
            User(student_id="DY20240001", institution_id="inst-001", role=UserRole.UNDERGRAD,
                 abac_attributes=ABACAttributes()),
            PasswordHasher.hash_password("demo123"),
        )
        l1_router.register_user(
            User(student_id="DY20240002", institution_id="inst-001", role=UserRole.TEACHER,
                 abac_attributes=ABACAttributes()),
            PasswordHasher.hash_password("demo123"),
        )
        # 研究生 (略懂) 与科研员 (熟悉) — 覆盖画像测试的中间能力档位
        l1_router.register_user(
            User(student_id="DY20240003", institution_id="inst-001", role=UserRole.GRADUATE,
                 abac_attributes=ABACAttributes()),
            PasswordHasher.hash_password("demo123"),
        )
        l1_router.register_user(
            User(student_id="DY20240004", institution_id="inst-001", role=UserRole.RESEARCHER,
                 abac_attributes=ABACAttributes()),
            PasswordHasher.hash_password("demo123"),
        )
        l1_router.register_user(
            User(student_id="DY20248888", institution_id="inst-001", role=UserRole.ADMIN,
                 abac_attributes=ABACAttributes()),
            PasswordHasher.hash_password("admin888"),
        )

        from dy3_polaris.l5.l1_gateway import L1AgentGateway

        l1_gateway = L1AgentGateway(l1_router)

        # L2 — 个性化层 (需要 IRT 服务)
        # 注: L2Router 由 UnifiedApp.__init__ 根据 irt_service 内部创建,
        # 此处仅初始化 IRT 服务即可。
        from dy3_polaris.l2.ability_assessor.tracing_service import IRTTracingService

        irt_service = IRTTracingService()

        # L2 画像/记忆全链路服务 (Knewton 三引擎 + Duolingo 实时学情)
        # 共享持久化 L2Store: BKT 追踪状态与画像掌握度同源, 重启不丢失学习数据
        from dy3_polaris.l2.knowledge_tracer.tracing_service import BKTTracingService
        from dy3_polaris.l2.memory.tracing_service import MemoryTracingService
        from dy3_polaris.l2.profile_builder.tracing_service import ProfileTracingService
        from dy3_polaris.l2.store import InMemoryL2Store

        # 数据目录: 生产默认源码 data; 测试经 data_dir 参数或 DY3_DATA_DIR
        # 环境变量隔离 (根治测试污染)
        if data_dir is None:
            data_dir = os.environ.get("DY3_DATA_DIR")
        data_root = Path(data_dir) if data_dir else (
            Path(__file__).resolve().parents[1]
        )
        _l2_data_dir = data_root / "l2" / "data" / "learners"
        l2_store = InMemoryL2Store(persist_dir=_l2_data_dir)
        bkt_service = BKTTracingService(store=l2_store)
        profile_service = ProfileTracingService(store=l2_store)
        memory_service = MemoryTracingService()

        # L3 — 知识层路由器
        from dy3_polaris.l3.store import KnowledgeStore
        from dy3_polaris.l3.concept_foundation import load_curated_concept_evidence
        from dy3_polaris.l3.api.router import L3Router
        from dy3_polaris.l3.fact_check import FactChecker
        from dy3_polaris.l3.quality_manager import QualityManager
        # 领域标准值库单点 (SSOT: shared/domain_standards.py)
        from dy3_polaris.shared.domain_standards import build_domain_standard_store

        # 事实校验器 (依赖标准值存储) 与 质量管理器 (L3Router 与决策引擎共享)
        standard_value_store = build_domain_standard_store()
        fact_checker = FactChecker(standard_value_store)
        quality_manager = QualityManager()

        l3_store = KnowledgeStore()
        # L3 持久化接入: 项目数据目录 + 启动加载上次快照 (P2 部署整改)
        from dy3_polaris.l3.persistence import PersistenceManager
        from dy3_polaris.l3.external_kb import build_external_knowledge_source

        # Knowledge assets and mutable learner/task data have different
        # lifecycles.  ``DY3_DATA_DIR`` is intentionally movable (tests,
        # deployments and user data isolation), while the reviewed domain
        # corpus shipped with the application is an immutable product asset.
        # Coupling both to ``data_dir`` previously made every isolated runtime
        # silently start with an empty knowledge base.
        l3_persist_base = (
            data_root / "l3" / "data" / "snapshots"
            if data_dir
            else _l3_snapshot_dir()
        )
        legacy_snapshot_override = os.environ.get("DY3_KNOWLEDGE_SNAPSHOT_DIR")
        canonical_l3_base = Path(legacy_snapshot_override or _l3_snapshot_dir())
        l3_persistence = PersistenceManager(l3_store, base_path=str(l3_persist_base))
        l3_persist_base.mkdir(parents=True, exist_ok=True)
        loaded_l3_snapshot: Path | None = None
        try:
            l3_persistence.load_snapshot()
            loaded_l3_snapshot = sorted(
                path for path in l3_persist_base.glob("snapshot_*") if path.is_dir()
            )[-1]
            _logger.info("L3 持久化快照已加载: %s", l3_persist_base)
        except Exception as exc:  # noqa: BLE001
            canonical_snapshots = (
                sorted(
                    path
                    for path in canonical_l3_base.glob("snapshot_*")
                    if path.is_dir()
                )
                if legacy_snapshot_override
                else [_l3_snapshot_path()]
            )
            if canonical_snapshots:
                try:
                    loaded_l3_snapshot = canonical_snapshots[-1]
                    l3_persistence.load_snapshot(loaded_l3_snapshot)
                    _logger.info(
                        "L3 canonical knowledge asset loaded: %s",
                        loaded_l3_snapshot,
                    )
                except Exception as canonical_exc:  # noqa: BLE001
                    _logger.warning(
                        "L3 canonical knowledge asset unavailable: %s",
                        canonical_exc,
                    )
            else:
                _logger.warning("L3 快照加载跳过 (首次启动): %s", exc)
        try:
            curated_concept_evidence = load_curated_concept_evidence(l3_store)
            _logger.info(
                "Reviewed Concept evidence loaded: %d records",
                len(curated_concept_evidence),
            )
        except Exception as exc:  # noqa: BLE001
            # A malformed governed asset must be visible in operations, but it
            # must not prevent the existing reviewed corpus from serving.
            _logger.error("Reviewed Concept evidence unavailable: %s", exc)
        # 嵌入管理器: 本地语义模型 (sentence-transformers, 离线) 供语义检索使用.
        # 模型惰性加载: 首次检索时才初始化, 不拖慢启动. 缺失时检索自动降级关键词.
        l3_embedding = None
        try:
            from dy3_polaris.l3.embedding import EmbeddingBackend, EmbeddingManager

            l3_embedding = EmbeddingManager(
                backend=EmbeddingBackend.SENTENCE_TRANSFORMERS,
                model_name=os.environ.get(
                    "DY3_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"
                ),
                normalize=True,
            )
            _logger.info("L3 语义检索已启用 (模型 %s, 惰性加载)", l3_embedding._model_name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("L3 嵌入管理器初始化失败, 语义检索降级关键词: %s", exc)
        # 向量索引: 加载离线构建的切片嵌入缓存 (bge-small-zh) 到内存向量索引,
        # 使混合检索的「向量」分支真正生效 (治检索不相关). 缓存缺失则向量检索
        # 自动降级为关键词检索, 不阻塞启动.
        if l3_embedding is not None:
            try:
                if loaded_l3_snapshot is not None:
                    _vec_file = loaded_l3_snapshot / "chunk_embeddings.json"
                    if _vec_file.exists():
                        import json as _json

                        _vecs = _json.loads(_vec_file.read_text(encoding="utf-8"))
                        _added = 0
                        for _cid, _vec in _vecs.items():
                            try:
                                l3_store.chunk_store.add_embedding(
                                    _cid, list(_vec), model="bge-small-zh-v1.5"
                                )
                                _added += 1
                            except Exception:  # noqa: BLE001
                                pass
                        _logger.info("向量索引已加载: %d 条切片嵌入", _added)
                    else:
                        _logger.info(
                            "未找到切片嵌入缓存 (chunk_embeddings.json), 向量检索待构建"
                        )
            except Exception as _exc:  # noqa: BLE001
                _logger.warning(
                    "向量索引加载失败, 关键词检索兜底: %s", type(_exc).__name__
                )
        l3_router = L3Router(
            l3_store,
            quality_manager=quality_manager,
            fact_checker=fact_checker,
            persistence_manager=l3_persistence,
            embedding_manager=l3_embedding,
        )

        # L4 — 决策引擎
        # 注: L4Router 由 UnifiedApp.__init__ 根据 decision_engine 内部创建,
        # 此处仅需构造 DecisionEngine 所需的依赖链即可。
        from dy3_polaris.l3.intent_router import IntentRouter
        from dy3_polaris.l3.graph_reasoner import GraphReasoner
        from dy3_polaris.l3.retrieval import HybridRetriever
        from dy3_polaris.l3.graphrag_retriever import GraphRAGRetriever
        from dy3_polaris.l4.decision_engine import DecisionEngine
        from dy3_polaris.l4.task_executor import TaskExecutor

        # T1: 意图路由器 (复用 L3 知识存储)
        intent_router = IntentRouter(l3_store)

        # L3 推理/检索组件 (TaskExecutor 依赖)
        graph_reasoner = GraphReasoner(l3_store)
        hybrid_retriever = HybridRetriever(l3_store)
        graphrag_retriever = GraphRAGRetriever(l3_store)

        # T3: 任务执行器
        task_executor = TaskExecutor(
            store=l3_store,
            graph_reasoner=graph_reasoner,
            hybrid_retriever=hybrid_retriever,
            graphrag_retriever=graphrag_retriever,
        )

        # 顶层决策引擎
        decision_engine = DecisionEngine(
            intent_router=intent_router,
            task_executor=task_executor,
            fact_checker=fact_checker,
            quality_manager=quality_manager,
        )

        # L5 — Agent Runtime (Temporal Workflow + LangGraph StateGraph 组合)
        from dy3_polaris.l3.reranker import (
            CompositeReranker,
            MMRReranker,
            QualityBoostReranker,
        )
        from dy3_polaris.l3.response_synthesizer import (
            ResponseSynthesizer,
            SynthesisConfig,
            SynthesisMode,
        )
        from dy3_polaris.l2.practice import PracticeBank
        from dy3_polaris.l5.agent_workers import AgentDependencies
        from dy3_polaris.l5.communication import MessageBus
        from dy3_polaris.l5.default_agents import build_default_agent_runtime
        from dy3_polaris.l5.orchestration_engine import OrchestrationEngine
        from dy3_polaris.l5.session_manager import SessionManager

        orchestration_engine = OrchestrationEngine()
        session_manager = SessionManager()
        raw_message_bus = MessageBus()
        # Outbox 可靠投递适配: publish 先入箱再投递 (P2 部署整改)
        from dy3_polaris.l5.outbox_bus import OutboxWiredBus

        message_bus = OutboxWiredBus(raw_message_bus)
        # 注册四个核心 Agent 的广播频道 (定义于 default_agents, 供跨 Agent 信息传播)
        from dy3_polaris.l5.default_agents import build_default_agents

        for definition in build_default_agents():
            for channel in definition.broadcast_channels:
                try:
                    message_bus.create_channel(channel.channel)
                except Exception:  # noqa: BLE001  频道已存在则跳过
                    pass
        reranker = CompositeReranker(
            [
                QualityBoostReranker(),
                MMRReranker(lambda_=0.75),
            ],
            top_k_per_stage=10,
        )
        from dy3_polaris.l2.user_understanding.service import UserUnderstandingService

        user_understanding_service = UserUnderstandingService(
            profile_store={},
            profile_service=profile_service,
        )
        agent_dependencies = AgentDependencies(
            irt_service=irt_service,
            profile_service=profile_service,
            memory_service=memory_service,
            bkt_service=bkt_service,
            practice_bank=PracticeBank(),
            message_bus=message_bus,
            l3_store=l3_store,
            hybrid_retriever=hybrid_retriever,
            graph_reasoner=graph_reasoner,
            graphrag_retriever=graphrag_retriever,
            fact_checker=fact_checker,
            quality_manager=quality_manager,
            anti_hallucination_pipeline=AntiHallucinationPipeline(),
            response_synthesizer=ResponseSynthesizer(
                SynthesisConfig(mode=SynthesisMode.REFINE, include_citations=True)
            ),
            reranker=reranker,
            decision_engine=decision_engine,
            audit_engine=l0_subsys.audit_engine,
            external_kb=build_external_knowledge_source(),
            embedding_manager=l3_embedding,
            user_understanding_service=user_understanding_service,
        )
        agent_runtime = build_default_agent_runtime(
            dependencies=agent_dependencies, message_bus=message_bus
        )

        # L6 — 协议基础设施
        from dy3_polaris.l6.core.engine import L6CoreEngine
        from dy3_polaris.l6.api.router import L6Router

        l6_engine = L6CoreEngine()
        l6_engine.initialize()
        # 注册工具目录 (internal/connector/skillbook/external), 修复"注册中心空转"
        try:
            from dy3_polaris.l6.registry import load_all_tools

            load_all_tools(l6_engine.tool_registry)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("L6 工具目录注册失败: %s", exc)
        l6_router = L6Router(l6_engine)

        # L7 — 体验呈现层 (T1-T6 全部渲染器/面板/API 规范)
        from dy3_polaris.l7.api.router import L7Router
        from dy3_polaris.l7.registry import get_registry
        from dy3_polaris.l7.renderers import register_native_renderers

        # 注册七大原生渲染器 (Text/Chart/Graph/Molecule/Table/Formula/Provenance)
        l7_registry = get_registry()
        registered_mimes = register_native_renderers(l7_registry)
        _logger.info("L7 native renderers registered: %d MIME types", len(registered_mimes))

        l7_router = L7Router(registry=l7_registry, api_prefix="/api/v1")

        # 演示数据必须显式开启，生产默认不伪造学习者经历。
        seed_demo_data = os.environ.get("DY3_SEED_DEMO_DATA", "0") == "1"
        if seed_demo_data:
            _seed_demo_learning_data(profile_service)
        # 知识库种子播种: 默认启用; 设置环境变量 DY3_SEED_KNOWLEDGE=0 可跳过,
        # 用于清空知识库后不再自动注入内置论文/资料 (用户可后续自行导入高质量文献)。
        if os.environ.get("DY3_SEED_KNOWLEDGE", "1") != "0":
            if seed_demo_data:
                _seed_demo_knowledge(l3_store, quality_manager)
            # L1-L4 层级知识图谱实体与关系播种 (供知识图谱分层可视化)
            _seed_domain_hierarchy(l3_store)
            # 领域知识播种: 8 篇稀土发光材料核心知识 (幂等), 使知识生成 Agent 可用
            from dy3_polaris.l3.knowledge_seed import seed_domain_knowledge

            seed_domain_knowledge(l3_store)
            # 知识点关系图播种 (镝-绿色健康照明垂直领域): 42 KP + 教学关系边 + 兜底事实入图
            # (前提/类比/因果/表征), 供问答沿知识点关系多跳拓展
            from dy3_polaris.l3.kp_graph_seed import seed_kp_graph

            seed_kp_graph(
                l3_store,
                include_placeholder_facts=(
                    os.environ.get("DY3_ENABLE_PLACEHOLDER_KNOWLEDGE", "0") == "1"
                ),
            )
            # 职业角色播种: 7 角色 + 角色-知识点关联 (多职业维度, 供个性化学习路径推荐)
            from dy3_polaris.l3.kp_graph_seed import seed_role_kp

            seed_role_kp(l3_store)
            # P1b 实体层 + Topic 层播种: 章/节主题节点 + 材料/离子/能级/方法/参数实体入图
            from dy3_polaris.l3.entity_topic_seed import seed_all as seed_entity_topic

            seed_entity_topic(l3_store)
            # P2 规则边补全: 材料/离子/参数/方法/主题之间的结构性关系 (measured_by/doped_with/...)
            from dy3_polaris.l3.edge_enrich import seed_structural_edges

            seed_structural_edges(l3_store)
            # P4 第 6 章新增知识点 + 应用主线边播种 (绿色健康照明应用)
            from dy3_polaris.l3.kp_graph_seed import seed_ch6_kps

            seed_ch6_kps(l3_store)
            # P2 LLM 补边: DeepSeek 生成 + 规则校验 + 人工抽查后的教学关系边 (applies_to/affects/...)
            from dy3_polaris.l3.edge_llm import seed_llm_edges

            seed_llm_edges(l3_store)
        else:
            _logger.info("DY3_SEED_KNOWLEDGE=0: 跳过知识库内置资料播种 (知识库为空, 等待导入)")

        # 组装全部八层 (L0-L7)
        app = cls(
            irt_service=irt_service,
            bkt_service=bkt_service,
            profile_service=profile_service,
            memory_service=memory_service,
            decision_engine=decision_engine,
            orchestration_engine=orchestration_engine,
            session_manager=session_manager,
            message_bus=message_bus,
            agent_runtime=agent_runtime,
            l1_gateway=l1_gateway,
            governance_router=governance_router,
            l1_router=l1_router,
            l3_router=l3_router,
            l6_router=l6_router,
            l7_router=l7_router,
            user_understanding_service=user_understanding_service,
            learning_task_store=task_state_runtime.LearningTaskStore(
                data_root / "l5" / "data" / "learning_tasks"
            ),
        )
        # 注入真实 Agent 依赖到技能执行器 (使其可动态调用)
        skill_executor = getattr(app, "_skill_executor", None)
        if skill_executor is not None:
            skill_executor.deps = agent_dependencies
        # Outbox 发件箱挂载到 bridge (跨层投递可见)
        try:
            app._bridge.outbox = message_bus.outbox  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        # L1 安全网关挂载到 bridge (供统一端点 Agent 审计)
        try:
            app._bridge.l1_gateway = l1_gateway  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return app


__all__ = ["UnifiedApp"]
