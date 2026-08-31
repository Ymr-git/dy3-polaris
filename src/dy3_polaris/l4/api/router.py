"""L4 决策引擎层 — REST API 路由层.

基于 Starlette 构建, 将 L4 决策引擎的核心功能暴露为 RESTful JSON API。

遵循与 L2/L3/L6 API 一致的设计模式:
- 统一响应格式: {"code": 0, "data": ..., "message": ""}
- CORS 中间件支持
- 异常统一处理
- 资源导向 URL 设计 (RESTful 语义)

融合世界先进方案的 API 设计:
- Knewton API: 决策引擎即服务 (query → action → confidence)
- LangGraph API: 状态化查询处理 (plan → execute → validate)
- TDP 框架: Supervisor-Planner-Executor 三层 API
- OLIVIA: 上下文线性赌博机 API (action selection)

端点列表:
- GET  /health:                     L4 健康检查
- POST /decision/query:             完整决策流程 (T1~T6)
- POST /decision/feedback:          提交反馈信号
- POST /decision/synthesize:        合成最终输出

使用示例::

    from dy3_polaris.l4.api import L4Router
    from dy3_polaris.l4 import DecisionEngine

    engine = DecisionEngine(intent_router=..., task_executor=...)
    router = L4Router(decision_engine=engine)
    app = router.create_app()

    # 独立运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

    # 或嵌入到主应用
    from starlette.routing import Mount
    main_routes = [Mount("/l4", app=router.create_app())]
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from dy3_polaris.l4.models import (
    ActionRecord,
    ActionType,
    ExecutionResult,
    ExecutionStatus,
    ValidationReport,
    ValidationSeverity,
)

_logger = logging.getLogger("dy3_polaris.l4.api.router")


# ============================================================
# 统一响应
# ============================================================


# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err, ok as _ok


def _safe_dump(obj: Any) -> Any:
    """安全地将 dataclass / dict / list 转为可 JSON 序列化的值."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value") and isinstance(obj, type):
        return obj.value
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _safe_dump(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ============================================================
# 路由处理器
# ============================================================


class _RouteHandlers:
    """将 L4 决策引擎方法适配为 Starlette Request→Response 处理器."""

    def __init__(self, decision_engine: Any, profile_service: Any = None) -> None:
        self._engine = decision_engine
        self._profile_service = profile_service

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /l4/health — L4 决策引擎健康检查."""
        services: dict[str, str] = {
            "decision_engine": "available",
            "validation": "available" if hasattr(self._engine, "_validation_orchestrator") and self._engine._validation_orchestrator is not None else "unavailable",
            "feedback": "available" if hasattr(self._engine, "_feedback_aggregator") and self._engine._feedback_aggregator is not None else "unavailable",
            "output_synthesis": "available" if hasattr(self._engine, "_output_synthesizer") and self._engine._output_synthesizer is not None else "unavailable",
        }
        return JSONResponse(_ok({
            "status": "healthy",
            "layer": "L4",
            "timestamp": time.time(),
            "services": services,
        }))

    # ---- 完整决策流程 (POST /l4/decision/query) ----

    async def decision_query(self, request: Request) -> JSONResponse:
        """POST /l4/decision/query — 完整决策流程 (T1~T6).

        请求体:
            query: 用户查询文本 (必填)
            context_id: 上下文 ID
            learner_profile: 学习者画像 (可选)
            query_vector: 预计算查询向量 (可选)

        响应:
            action_type: 行动类型
            confidence: 置信度
            response_payload: 响应载荷
            plan_id: 计划 ID
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        query = body.get("query")
        if not query:
            return JSONResponse(_err(-32700, "缺少必填参数: query"), status_code=400)

        context_id = body.get("context_id", "")
        learner_profile = body.get("learner_profile")
        query_vector = body.get("query_vector")

        try:
            action_record = await self._engine.process_query(
                query=query,
                context_id=context_id,
                learner_profile=learner_profile,
                query_vector=query_vector,
            )
            return JSONResponse(_ok(_safe_dump(action_record)))
        except Exception as e:
            _logger.exception("决策引擎处理失败")
            return JSONResponse(_err(-32400, "决策处理失败", str(e)), status_code=500)

    # ---- 学习策略决策 (POST /l4/decision/next-action) ----

    async def decision_next_action(self, request: Request) -> JSONResponse:
        """POST /l4/decision/next-action — 唯一策略决策点.

        请求体:
            learner_id: 学习者 ID (必填)
            mode: 策略模式 (default/review/guide/assess, 默认 default)
            learner_profile: 学习者画像 (可选, 缺省时由 L5 组装传入)
            context_id: 上下文 ID (L1 会话 ID)

        响应 data (统一决策语义):
            action_type: review/practice/assess/learn
            confidence: 决策置信度
            recommended_path: [{kp_id, action, target, effort}] 推荐 KP 步骤
            plan_id / mode / summary / learner_id / context_id
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id")
        if not learner_id:
            return JSONResponse(_err(-32700, "缺少必填参数: learner_id"), status_code=400)

        mode = str(body.get("mode") or "default")
        if mode not in ("default", "review", "guide", "assess"):
            return JSONResponse(
                _err(-32602, f"非法策略模式: {mode} (可选 default/review/guide/assess)"),
                status_code=400,
            )

        try:
            # 画像自足: 调用方未传 learner_profile 时, 由 L4 自行从 L2 拉取
            learner_profile = body.get("learner_profile") or None
            if learner_profile is None and self._profile_service is not None:
                try:
                    profile = self._profile_service.get_profile_snapshot(learner_id)
                    if profile is not None and hasattr(profile, "to_dict"):
                        learner_profile = profile.to_dict()
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("L4 自拉画像失败 %s: %s", learner_id, exc)
            decision = await self._engine.process_next_action(
                learner_id,
                mode=mode,
                learner_profile=learner_profile,
                context_id=str(body.get("context_id") or ""),
            )
            return JSONResponse(_ok(decision))
        except Exception as e:
            _logger.exception("学习策略决策失败")
            return JSONResponse(_err(-32400, "学习策略决策失败", str(e)), status_code=500)

    # ---- 提交反馈 (POST /l4/decision/feedback) ----

    async def decision_feedback(self, request: Request) -> JSONResponse:
        """POST /l4/decision/feedback — 提交反馈信号.

        请求体:
            action_id: 行动 ID (必填)
            feedback_type: 反馈类型 (explicit/implicit)
            rating: 评分 [0, 1]
            comment: 评论文本 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        action_id = body.get("action_id")
        if not action_id:
            return JSONResponse(_err(-32700, "缺少必填参数: action_id"), status_code=400)

        feedback_type = body.get("feedback_type", "explicit_rating")
        rating = float(body.get("rating", 0.5))
        comment = body.get("comment", "")

        try:
            # 调用决策引擎的反馈记录方法 (构造 ActionRecord 匹配实现签名)
            if hasattr(self._engine, "record_feedback"):
                from dy3_polaris.l4.models import ActionRecord, ActionType, FeedbackType

                # 归一化反馈类型 (兼容 'explicit' 等别名)
                _fb_map = {
                    "explicit": FeedbackType.EXPLICIT_RATING,
                    "implicit": FeedbackType.IMPLICIT_SIGNAL,
                    "outcome": FeedbackType.OUTCOME_FEEDBACK,
                }
                try:
                    fb_enum = FeedbackType(feedback_type)
                except ValueError:
                    fb_enum = _fb_map.get(feedback_type, FeedbackType.EXPLICIT_RATING)

                record = ActionRecord(
                    record_id=action_id,
                    plan_id=body.get("plan_id", ""),
                    action_type=ActionType(body.get("action_type", "direct_answer"))
                    if body.get("action_type") else ActionType.DIRECT_ANSWER,
                    confidence=float(body.get("confidence", 0.5)),
                )
                self._engine.record_feedback(
                    action_record=record,
                    rating=rating,
                    comment=comment,
                    feedback_type=fb_enum,
                    intent_type=body.get("intent_type", ""),
                )
            elif hasattr(self._engine, "_feedback_aggregator") and self._engine._feedback_aggregator:
                self._engine._feedback_aggregator.record_feedback(
                    action_id=action_id,
                    feedback_type=feedback_type,
                    rating=rating,
                )

            return JSONResponse(_ok({"action_id": action_id, "recorded": True}))
        except Exception as e:
            _logger.exception("反馈记录失败")
            return JSONResponse(_err(-32400, "反馈记录失败", str(e)), status_code=500)

    # ---- 合成最终输出 (POST /l4/decision/synthesize) ----

    async def decision_synthesize(self, request: Request) -> JSONResponse:
        """POST /l4/decision/synthesize — 合成最终输出 (T5+).

        请求体: ActionRecord + ExecutionResult + ValidationReport 关键字段
        响应: OutputRecord (content/confidence/safety_level)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            plan_id = str(body.get("plan_id") or "")
            action_value = str(body.get("action_type") or "direct_answer")
            # 兼容旧路由的 respond 命名，内部始终构造正式模型。
            if action_value == "respond":
                action_value = ActionType.DIRECT_ANSWER.value
            validation_value = str(body.get("validation_status") or "pass")
            if validation_value == "passed":
                validation_value = ValidationSeverity.PASS.value

            action_record = ActionRecord(
                plan_id=plan_id,
                action_type=ActionType(action_value),
                confidence=float(body.get("confidence", 0.5)),
                execution_confidence=float(
                    body.get("execution_confidence", body.get("confidence", 0.5))
                ),
                validation_score=float(body.get("validation_score", 0.5)),
                response_payload=dict(body.get("response_payload") or {}),
            )
            execution_result = ExecutionResult(
                plan_id=plan_id,
                status=ExecutionStatus(
                    str(body.get("execution_status") or ExecutionStatus.COMPLETED.value)
                ),
                confidence=float(
                    body.get("execution_confidence", body.get("confidence", 0.5))
                ),
            )
            validation_report = ValidationReport(
                plan_id=plan_id,
                overall_status=ValidationSeverity(validation_value),
                overall_score=float(body.get("validation_score", 0.5)),
                refinement_iterations=int(body.get("refinement_iterations", 0)),
            )

            output = self._engine.synthesize_output(
                action_record,
                execution_result,
                validation_report,
            )

            if output is not None:
                return JSONResponse(_ok(_safe_dump(output)))
            else:
                return JSONResponse(_ok({
                    "content": body.get("response_payload", {}).get("answer", ""),
                    "confidence": body.get("confidence", 0.5),
                    "safety_level": "safe",
                }))
        except Exception as e:
            _logger.exception("输出合成失败")
            return JSONResponse(_err(-32400, "输出合成失败", str(e)), status_code=500)


# ============================================================
# L4Router
# ============================================================


class L4Router:
    """L4 决策引擎 REST API 路由器.

    将 DecisionEngine 的核心功能暴露为 RESTful API。
    遵循与 L2Router / L3Router / L6Router 一致的设计模式。

    使用示例::

        from dy3_polaris.l4.api import L4Router
        from dy3_polaris.l4 import DecisionEngine

        engine = DecisionEngine(intent_router=..., task_executor=...)
        router = L4Router(decision_engine=engine)
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8004)

        # 或嵌入到主应用
        from starlette.routing import Mount
        main_routes = [Mount("/l4", app=router.create_app())]

    Args:
        decision_engine: L4 决策引擎实例 (必填).
        cors_origins: CORS 允许的源 (默认 ["*"]).
    """

    def __init__(
        self,
        decision_engine: Any,
        cors_origins: list[str] | None = None,
        profile_service: Any = None,
    ) -> None:
        """初始化 L4 路由器.

        Args:
            decision_engine: L4 决策引擎实例 (必填).
            cors_origins: CORS 允许的源 (默认 ["*"]).
            profile_service: L2 画像服务 (可选; 缺失时 next-action 依赖调用方传画像).
        """
        self._engine = decision_engine
        self._cors_origins = cors_origins or ["*"]
        self._profile_service = profile_service
        self._handlers = _RouteHandlers(
            decision_engine=decision_engine, profile_service=profile_service
        )

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        Returns:
            配置好的 Starlette 应用, 可直接传给 uvicorn.run()
            或通过 Mount 嵌入到主应用。
        """
        h = self._handlers

        routes = [
            # 健康检查
            Route("/health", h.health, methods=["GET"]),

            # 决策引擎
            Route("/decision/query", h.decision_query, methods=["POST"]),
            Route("/decision/feedback", h.decision_feedback, methods=["POST"]),
            Route("/decision/synthesize", h.decision_synthesize, methods=["POST"]),
            # 学习策略决策 (唯一策略决策点)
            Route("/decision/next-action", h.decision_next_action, methods=["POST"]),
        ]

        middleware = []
        if self._cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._cors_origins,
                    allow_methods=["*"] if "*" in self._cors_origins
                                   else ["GET", "POST", "PUT", "DELETE"],
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)
        return app

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取所有路由摘要 (用于文档/发现).

        Returns:
            [{"path": ..., "methods": [...], "description": ...}]
        """
        return [
            {"path": "/health", "methods": ["GET"], "description": "L4 决策引擎健康检查"},
            {"path": "/decision/query", "methods": ["POST"], "description": "完整决策流程 (T1~T6)"},
            {"path": "/decision/feedback", "methods": ["POST"], "description": "提交反馈信号"},
            {"path": "/decision/synthesize", "methods": ["POST"], "description": "合成最终输出"},
        ]


__all__ = ["L4Router"]
