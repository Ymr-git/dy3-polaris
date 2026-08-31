"""L5 Agent Runtime 层 — REST API 路由层.

基于 Starlette 构建, 将 L5 编排引擎、会话管理、消息总线等核心功能
暴露为 RESTful JSON API。

遵循与 L2/L3/L4/L6 API 一致的设计模式:
- 统一响应格式: {"code": 0, "data": ..., "message": ""}
- CORS 中间件支持
- 异常统一处理
- 资源导向 URL 设计 (RESTful 语义)

融合世界先进方案的 API 设计:
- LangGraph API: StateGraph 编排 + checkpoint 恢复
- Temporal API: Workflow/Activity 管理 + 事件溯源
- Google ADK API: Session/State/Memory 三层管理
- OpenAI Agents SDK API: Handoff + Guardrail
- AutoGen API: GroupChat Manager + Topic 发布订阅

端点列表:
- GET  /health:                              L5 健康检查
- POST /orchestrate:                         执行编排计划 (Pipeline/Debate/Voting)
- GET  /orchestration/{plan_id}:             获取编排结果
- POST /session:                             创建会话
- GET  /session/{session_id}:                获取会话状态
- POST /session/{session_id}/fork:           Fork 会话
- POST /session/{session_id}/close:          关闭会话
- POST /message/publish:                     发布消息到总线

使用示例::

    from dy3_polaris.l5.api import L5Router
    from dy3_polaris.l5 import OrchestrationEngine, SessionManager, MessageBus

    router = L5Router(
        orchestration_engine=OrchestrationEngine(),
        session_manager=SessionManager(),
        message_bus=MessageBus(),
    )
    app = router.create_app()

    # 独立运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

    # 或嵌入到主应用
    from starlette.routing import Mount
    main_routes = [Mount("/l5", app=router.create_app())]
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

_logger = logging.getLogger("dy3_polaris.l5.api.router")


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
    """将 L5 Agent Runtime 方法适配为 Starlette Request→Response 处理器."""

    def __init__(
        self,
        orchestration_engine: Any,
        session_manager: Any,
        message_bus: Any,
        agent_runtime: Any | None = None,
        l1_gateway: Any | None = None,
        skill_executor: Any | None = None,
    ) -> None:
        self._orch = orchestration_engine
        self._session = session_manager
        self._bus = message_bus
        self._agents = agent_runtime
        self._l1 = l1_gateway
        self._skills = skill_executor

    # ---- 技能 (Skill) 端点 ----

    async def skills(self, request: Request) -> JSONResponse:
        """GET /l5/agents/{agent_id}/skills — 列出 Agent 可用技能."""
        if self._skills is None:
            return JSONResponse(_err(-32401, "技能执行器未初始化"), status_code=503)
        agent_id = request.path_params.get("agent_id", "")
        try:
            return JSONResponse(_ok({
                "agent_id": agent_id,
                "skills": self._skills.list_skills(agent_id or None),
            }))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(_err(-32400, "技能列表失败", str(e)), status_code=500)

    async def call_skill(self, request: Request) -> JSONResponse:
        """POST /l5/agents/{agent_id}/skills/call — 动态调用 Agent 技能.

        请求体:
            tool: 技能名 (全名 internal.bkt_compute 或短名 bkt_compute)
            args: 技能参数
        """
        if self._skills is None:
            return JSONResponse(_err(-32401, "技能执行器未初始化"), status_code=503)
        agent_id = request.path_params.get("agent_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}
        tool = body.get("tool", "")
        if not tool:
            return JSONResponse(_err(-32700, "缺少必填参数: tool"), status_code=400)
        try:
            result = self._skills.call(agent_id, tool, body.get("args") or {})
            return JSONResponse(_ok(result))
        except KeyError as e:
            return JSONResponse(_err(-32602, str(e)), status_code=404)
        except ValueError as e:
            return JSONResponse(_err(-32203, str(e)), status_code=403)
        except Exception as e:  # noqa: BLE001
            _logger.exception("技能执行失败")
            return JSONResponse(_err(-32400, "技能执行失败", str(e)), status_code=500)

    # ---- Agent 广播 inbox (按需协作消费) ----

    async def agent_inbox(self, request: Request) -> JSONResponse:
        """GET /l5/agents/{agent_id}/inbox — 返回 Agent 收到的广播消息.

        基于 MessageBus 订阅 (bind_message_bus 注册 SUB 频道回调),
        Agent 可按需消费其他 Agent 广播的信息 (诊断→生成/审核/决策 等).
        """
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        agent_id = request.path_params.get("agent_id", "")
        try:
            inbox = self._agents.get_inbox(agent_id)
            return JSONResponse(_ok({"agent_id": agent_id, "total": len(inbox), "messages": inbox}))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(_err(-32400, "Inbox 读取失败", str(e)), status_code=500)

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /l5/health — L5 Agent Runtime 健康检查."""
        services: dict[str, str] = {
            "orchestration": "available" if self._orch is not None else "unavailable",
            "session": "available" if self._session is not None else "unavailable",
            "message_bus": "available" if self._bus is not None else "unavailable",
            "agent_runtime": "available" if self._agents is not None else "unavailable",
        }
        return JSONResponse(_ok({
            "status": "healthy",
            "layer": "L5",
            "timestamp": time.time(),
            "services": services,
        }))

    # ---- Agent 运行时状态 (GET /l5/agents) ----

    async def agents(self, request: Request) -> JSONResponse:
        """GET /l5/agents — 列出已注册 Agent 与实例状态."""
        if self._agents is None:
            return JSONResponse(_ok({
                "available": False,
                "total": 0,
                "agents": [],
            }))
        try:
            status = await self._agents.list_status()
            return JSONResponse(_ok(status))
        except Exception as e:  # noqa: BLE001
            _logger.exception("Agent 状态读取失败")
            return JSONResponse(
                _err(-32400, "Agent 状态读取失败", str(e)),
                status_code=500,
            )

    async def agent_detail(self, request: Request) -> JSONResponse:
        """GET /l5/agents/{agent_id} — 获取单个 Agent 状态."""
        if self._agents is None:
            return JSONResponse(_err(-32400, "Agent 运行时未初始化"), status_code=503)
        agent_id = request.path_params.get("agent_id", "")
        try:
            definition = self._agents.registry.get(agent_id)
            if definition is None:
                return JSONResponse(_err(-32600, "Agent 不存在"), status_code=404)
            await self._agents.ensure_instances([agent_id])
            item = definition.to_dict()
            instance = self._agents.get_instance(agent_id)
            item["instance"] = instance.health_check() if instance else None
            return JSONResponse(_ok(item))
        except Exception as e:  # noqa: BLE001
            _logger.exception("Agent 详情失败")
            return JSONResponse(_err(-32400, "Agent 详情失败", str(e)), status_code=500)

    async def run_agent(self, request: Request) -> JSONResponse:
        """POST /l5/agents/{agent_id}/run — 执行单个 Agent worker."""
        if self._agents is None:
            return JSONResponse(_err(-32400, "Agent 运行时未初始化"), status_code=503)
        agent_id = request.path_params.get("agent_id", "")
        if self._l1 is not None:
            user = self._l1.authenticate(request)
            if user is None:
                return JSONResponse(
                    _err(-32201, "AUTHENTICATION_ERROR", "未认证"),
                    status_code=401,
                )
            allowed, reason = self._l1.check_agent_permission(user, agent_id)
            if not allowed:
                self._l1.audit_agent_call(
                    user, agent_id, success=False, detail=reason
                )
                return JSONResponse(
                    _err(-32203, "FORBIDDEN", reason),
                    status_code=403,
                )
        if self._agents.registry.get(agent_id) is None:
            return JSONResponse(_err(-32600, "Agent 不存在"), status_code=404)
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            # 超时兜底: worker 挂起时 30s 内返回降级结果, 避免请求永久悬挂
            timeout_s = float(body.get("timeout_s", 30.0))
            try:
                result = await asyncio.wait_for(
                    self._agents.run(agent_id, body), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                _logger.warning("Agent %s 执行超时 (%.1fs), 返回降级结果", agent_id, timeout_s)
                result = {
                    "agent_id": agent_id,
                    "status": "timeout",
                    "timeout_s": timeout_s,
                    "summary": f"Agent 执行超过 {timeout_s:.0f}s，已超时降级（可重试）",
                    "fallback": True,
                }
            if self._l1 is not None and user is not None:
                self._l1.audit_agent_call(
                    user,
                    agent_id,
                    success=True,
                    detail=result.get("status", "completed"),
                )
                self._l1.collect_agent_call(
                    user.user_id,
                    str(body.get("session_id", "") or ""),
                    agent_id,
                    str(
                        result.get("summary")
                        or result.get("answer")
                        or result.get("reason")
                        or ""
                    ),
                )
            return JSONResponse(_ok(result))
        except Exception as e:  # noqa: BLE001
            _logger.exception("Agent 执行失败")
            return JSONResponse(_err(-32400, "Agent 执行失败", str(e)), status_code=500)

    # ---- 执行编排计划 (POST /l5/orchestrate) ----

    async def orchestrate(self, request: Request) -> JSONResponse:
        """POST /l5/orchestrate — 执行编排计划 (Pipeline/Debate/Voting).

        请求体:
            paradigm: 编排范式 (pipeline/debate/voting) (必填)
            tasks: 任务列表 [{task_id, agent_id, input}]
            session_id: 会话 ID (可选)
            config: 编排配置 (可选)

        响应:
            plan_id: 编排计划 ID
            state: 编排状态 (completed/failed/timeout)
            results: 各任务执行结果
            elapsed_ms: 总耗时
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        paradigm = body.get("paradigm")
        if not paradigm:
            return JSONResponse(_err(-32700, "缺少必填参数: paradigm"), status_code=400)

        tasks = body.get("tasks", [])
        session_id = body.get("session_id", "")
        config = body.get("config", {})

        try:
            # 构建编排计划 (OrchestrationPlan + 任务列表)
            from dy3_polaris.l5.orchestration_engine import (
                OrchestrationParadigm,
                OrchestrationPlan,
                OrchestrationTask,
            )

            try:
                paradigm_enum = OrchestrationParadigm(paradigm)
            except ValueError:
                return JSONResponse(
                    _err(-32602, f"未知编排范式: {paradigm}"), status_code=400
                )

            plan_id = str(uuid.uuid4())
            plan_tasks: list[OrchestrationTask] = []
            for t in tasks:
                task = OrchestrationTask(
                    task_id=t.get("task_id", f"t{len(plan_tasks)}"),
                    agent_id=t.get("agent_id", ""),
                    name=t.get("name", ""),
                    dependencies=t.get("dependencies") or [],
                    timeout_s=float(t.get("timeout_s", 120.0)),
                )
                task.input_data = dict(t.get("input") or {})
                plan_tasks.append(task)

            plan = OrchestrationPlan(
                plan_id=plan_id,
                tasks=plan_tasks,
                paradigm=paradigm_enum,
                total_timeout_s=float(config.get("total_timeout_s", 60.0))
                if isinstance(config, dict) else 60.0,
            )

            async def execute_fn(
                task: OrchestrationTask, _context: dict[str, Any]
            ) -> dict[str, Any]:
                """按任务 agent_id 调用 Agent 运行时 worker."""
                if self._agents is None:
                    raise RuntimeError("Agent 运行时未初始化")
                return await self._agents.run(task.agent_id, task.input_data)

            # 调用编排引擎执行
            result = await self._orch.execute(plan, execute_fn)
            return JSONResponse(_ok(_safe_dump(result)))
        except Exception as e:
            _logger.exception("编排执行失败")
            return JSONResponse(_err(-32400, "编排执行失败", str(e)), status_code=500)

    # ---- 获取编排结果 (GET /l5/orchestration/{plan_id}) ----

    async def get_orchestration_result(self, request: Request) -> JSONResponse:
        """GET /l5/orchestration/{plan_id} — 获取编排结果."""
        plan_id = request.path_params.get("plan_id", "")
        if not plan_id:
            return JSONResponse(_err(-32700, "缺少路径参数: plan_id"), status_code=400)

        try:
            result = self._orch.get_result(plan_id)
            if result is None:
                return JSONResponse(_err(-32600, "编排结果不存在"), status_code=404)
            return JSONResponse(_ok(_safe_dump(result)))
        except Exception as e:
            _logger.exception("获取编排结果失败")
            return JSONResponse(_err(-32400, "获取结果失败", str(e)), status_code=500)

    # ---- 创建会话 (POST /l5/session) ----

    async def create_session(self, request: Request) -> JSONResponse:
        """POST /l5/session — 创建会话.

        请求体:
            learner_id: 学习者 ID
            context: 会话上下文 (可选)
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        learner_id = body.get("learner_id", "")
        context = body.get("context", {})
        agent_id = (
            body.get("agent_id")
            or (context.get("agent_id") if isinstance(context, dict) else None)
            or "main"
        )

        try:
            session = self._session.create_session(
                agent_id=agent_id,
                learner_id=learner_id,
            )
            return JSONResponse(_ok(_safe_dump(session)))
        except Exception as e:
            _logger.exception("创建会话失败")
            return JSONResponse(_err(-32400, "创建会话失败", str(e)), status_code=500)

    # ---- 获取会话状态 (GET /l5/session/{session_id}) ----

    async def get_session(self, request: Request) -> JSONResponse:
        """GET /l5/session/{session_id} — 获取会话状态."""
        session_id = request.path_params.get("session_id", "")
        if not session_id:
            return JSONResponse(_err(-32700, "缺少路径参数: session_id"), status_code=400)

        try:
            session = self._session.get_session(session_id)
            if session is None:
                return JSONResponse(_err(-32600, "会话不存在"), status_code=404)
            return JSONResponse(_ok(_safe_dump(session)))
        except Exception as e:
            _logger.exception("获取会话失败")
            return JSONResponse(_err(-32400, "获取会话失败", str(e)), status_code=500)

    # ---- Fork 会话 (POST /l5/session/{session_id}/fork) ----

    async def fork_session(self, request: Request) -> JSONResponse:
        """POST /l5/session/{session_id}/fork — Fork 会话.

        请求体:
            fork_reason: Fork 原因 (可选)
        """
        session_id = request.path_params.get("session_id", "")
        if not session_id:
            return JSONResponse(_err(-32700, "缺少路径参数: session_id"), status_code=400)

        try:
            body = await request.json()
        except Exception:
            body = {}

        fork_reason = body.get("fork_reason", "manual")

        try:
            forked = self._session.create_fork(
                session_id=session_id,
                trigger_type=body.get("trigger_type", "manual"),
                initiator=body.get("initiator", "user"),
                reason=fork_reason,
            )
            return JSONResponse(_ok(_safe_dump(forked)))
        except Exception as e:
            _logger.exception("Fork 会话失败")
            return JSONResponse(_err(-32400, "Fork 会话失败", str(e)), status_code=500)

    # ---- 关闭会话 (POST /l5/session/{session_id}/close) ----

    async def close_session(self, request: Request) -> JSONResponse:
        """POST /l5/session/{session_id}/close — 关闭会话."""
        session_id = request.path_params.get("session_id", "")
        if not session_id:
            return JSONResponse(_err(-32700, "缺少路径参数: session_id"), status_code=400)

        try:
            success = self._session.close(session_id)
            return JSONResponse(_ok({"session_id": session_id, "closed": success}))
        except Exception as e:
            _logger.exception("关闭会话失败")
            return JSONResponse(_err(-32400, "关闭会话失败", str(e)), status_code=500)

    # ---- 交互记录 (GET /l5/interactions) ----

    async def interaction_stats(self, request: Request) -> JSONResponse:
        """GET /l5/interactions/stats — 获取交互记录统计."""
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        try:
            recorder = self._agents.get_recorder()
            stats = recorder.get_stats()
            return JSONResponse(_ok(stats))
        except Exception as e:
            _logger.exception("交互统计失败")
            return JSONResponse(_err(-32400, "交互统计失败", str(e)), status_code=500)

    async def interaction_chains(self, request: Request) -> JSONResponse:
        """GET /l5/interactions/chains — 获取交互链列表."""
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        try:
            limit = int(request.query_params.get("limit", "20"))
            offset = int(request.query_params.get("offset", "0"))
            recorder = self._agents.get_recorder()
            chains = recorder.get_all_chains(limit=limit, offset=offset)
            return JSONResponse(_ok({
                "total": len(chains),
                "chains": chains,
            }))
        except Exception as e:
            _logger.exception("交互链列表失败")
            return JSONResponse(_err(-32400, "交互链列表失败", str(e)), status_code=500)

    async def interaction_chain_detail(self, request: Request) -> JSONResponse:
        """GET /l5/interactions/chains/{chain_id} — 获取交互链详情."""
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        chain_id = request.path_params.get("chain_id", "")
        if not chain_id:
            return JSONResponse(_err(-32700, "缺少路径参数: chain_id"), status_code=400)
        try:
            recorder = self._agents.get_recorder()
            chain = recorder.get_chain_detail(chain_id)
            if chain is None:
                return JSONResponse(_err(-32600, "交互链不存在"), status_code=404)
            return JSONResponse(_ok(chain))
        except Exception as e:
            _logger.exception("交互链详情失败")
            return JSONResponse(_err(-32400, "交互链详情失败", str(e)), status_code=500)

    async def interaction_records(self, request: Request) -> JSONResponse:
        """GET /l5/interactions/records — 获取交互记录列表."""
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        try:
            limit = int(request.query_params.get("limit", "50"))
            offset = int(request.query_params.get("offset", "0"))
            agent_id = request.query_params.get("agent_id", "")
            phase = request.query_params.get("phase", "")
            recorder = self._agents.get_recorder()
            records = recorder.get_all_records(
                limit=limit, offset=offset, agent_id=agent_id, phase=phase
            )
            return JSONResponse(_ok({
                "total": len(records),
                "records": records,
                "stats": recorder.get_stats(),
            }))
        except Exception as e:
            _logger.exception("交互记录列表失败")
            return JSONResponse(_err(-32400, "交互记录列表失败", str(e)), status_code=500)

    async def interaction_latest(self, request: Request) -> JSONResponse:
        """GET /l5/interactions/latest — 获取最新交互记录."""
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        try:
            limit = int(request.query_params.get("limit", "30"))
            recorder = self._agents.get_recorder()
            records = recorder.get_latest_records(limit=limit)
            stats = recorder.get_stats()
            return JSONResponse(_ok({
                "records": records,
                "stats": stats,
            }))
        except Exception as e:
            _logger.exception("最新交互记录失败")
            return JSONResponse(_err(-32400, "最新交互记录失败", str(e)), status_code=500)

    async def interaction_demo(self, request: Request) -> JSONResponse:
        """POST /l5/interactions/demo — 生成演示交互数据.

        生成 5 条完整的 4-Agent 协作交互链（诊断→生成→审核→决策），
        用于前端监控页面展示效果，无需实际发起多智能体交互即可查看轨迹。
        """
        if self._agents is None:
            return JSONResponse(_err(-32401, "Agent 运行时未初始化"), status_code=503)
        try:
            recorder = self._agents.get_recorder()
            demo_queries = [
                "Dy3+的浓度猝灭机理是什么？",
                "Eu3+的发光效率与温度关系如何？",
                "Ce3+掺杂的YAG荧光粉量子效率优化",
                "Tb3+绿色荧光粉的制备方法",
                "上转换发光材料的能量传递机制",
            ]
            demo_agents = [
                ("agent.learning.diagnosis", "学情诊断", InteractionPhase.DIAGNOSIS),
                ("agent.knowledge.generation", "知识生成", InteractionPhase.GENERATION),
                ("agent.quality.review", "审核校验", InteractionPhase.REVIEW),
                ("agent.guidance.decision", "导学决策", InteractionPhase.DECISION),
            ]
            demo_actions = {
                "diagnosis": ["分析学习者画像", "检索薄弱知识点", "识别知识盲区", "评估当前掌握度"],
                "generation": ["生成知识解释", "构建学习内容", "组织知识图谱", "编写示例说明"],
                "review": ["校验事实准确性", "检测逻辑一致性", "验证引用来源", "评估内容质量"],
                "decision": ["制定学习路径", "推荐练习策略", "规划复习计划", "生成学习建议"],
            }
            import random
            random.seed(time.time())
            count = 0
            for q in demo_queries:
                chain_id = recorder.start_chain(
                    session_id=f"demo_session_{count}",
                    learner_id="DY20240001",
                    query=q,
                )
                agents = list(demo_agents)
                random.shuffle(agents)
                for aid, aname, phase in agents:
                    actions = demo_actions.get(phase.value, ["执行任务"])
                    action = random.choice(actions)
                    dur = random.uniform(500, 5000)
                    status = "completed" if random.random() > 0.15 else "failed"
                    recorder.record_agent_execution(
                        agent_id=aid,
                        agent_name=aname,
                        action=action,
                        input_data={"query": q, "context": {"phase": phase.value}},
                        output_data={"summary": f"{aname}完成: {action}"},
                        duration_ms=dur,
                        status=status,
                        phase=phase,
                        chain_id=chain_id,
                    )
                recorder.end_chain(
                    chain_id=chain_id,
                    final_answer=f"关于「{q}」的完整分析已生成，包含诊断结果、知识解释、审核结论和学习建议。",
                    status="completed",
                )
                count += 1
            return JSONResponse(_ok({
                "generated": count,
                "message": f"已生成 {count} 条演示交互链，请刷新页面查看。",
            }))
        except Exception as e:
            _logger.exception("生成演示数据失败")
            return JSONResponse(_err(-32400, "生成演示数据失败", str(e)), status_code=500)

    # ---- 发布消息 (POST /l5/message/publish) ----

    async def publish_message(self, request: Request) -> JSONResponse:
        """POST /l5/message/publish — 发布消息到总线.

        请求体:
            channel: 频道名称 (必填)
            payload: 消息载荷 (必填)
            publisher: 发布者 Agent ID (默认 "external")
            priority: 消息优先级 (low/normal/high)
        """
        if self._bus is None:
            return JSONResponse(_err(-32401, "消息总线不可用"), status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        channel = body.get("channel")
        if not channel:
            return JSONResponse(_err(-32700, "缺少必填参数: channel"), status_code=400)

        payload = body.get("payload", {})
        publisher = body.get("publisher", "external")
        priority = body.get("priority", "normal")

        try:
            # MessageBus.publish 需要 Message 对象 (频道须已注册)
            from dy3_polaris.l5.communication import Message, MessagePriority

            try:
                prio = MessagePriority(priority)
            except ValueError:
                prio = MessagePriority.NORMAL
            msg = Message(
                channel=channel,
                publisher=str(publisher),
                payload=payload,
                priority=prio,
            )
            self._bus.publish(msg)
            return JSONResponse(_ok({
                "message_id": msg.message_id,
                "channel": channel,
                "stream_id": msg.stream_id,
            }))
        except Exception as e:
            _logger.exception("消息发布失败")
            return JSONResponse(_err(-32400, "消息发布失败", str(e)), status_code=500)


# ============================================================
# L5Router
# ============================================================


class L5Router:
    """L5 Agent Runtime REST API 路由器.

    将 OrchestrationEngine / SessionManager / MessageBus 的核心功能
    暴露为 RESTful API。遵循与 L2Router / L3Router / L4Router 一致的设计模式。

    使用示例::

        from dy3_polaris.l5.api import L5Router
        from dy3_polaris.l5 import OrchestrationEngine, SessionManager, MessageBus

        router = L5Router(
            orchestration_engine=OrchestrationEngine(),
            session_manager=SessionManager(),
            message_bus=MessageBus(),
        )
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8005)

        # 或嵌入到主应用
        from starlette.routing import Mount
        main_routes = [Mount("/l5", app=router.create_app())]

    Args:
        orchestration_engine: L5 编排引擎 (可选).
        session_manager: L5 会话管理器 (可选).
        message_bus: L5 消息总线 (可选).
        agent_runtime: 默认 Agent 运行时 (可选).
        l1_gateway: L1 Agent 安全网关 (可选).
        cors_origins: CORS 允许的源 (默认 ["*"]).
    """

    def __init__(
        self,
        orchestration_engine: Any | None = None,
        session_manager: Any | None = None,
        message_bus: Any | None = None,
        agent_runtime: Any | None = None,
        l1_gateway: Any | None = None,
        skill_executor: Any | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        """初始化 L5 路由器.

        Args:
            orchestration_engine: L5 编排引擎 (可选).
            session_manager: L5 会话管理器 (可选).
            message_bus: L5 消息总线 (可选).
            agent_runtime: 默认 Agent 运行时 (可选).
            l1_gateway: L1 Agent 安全网关 (可选).
            skill_executor: L5 技能执行器 (可选).
            cors_origins: CORS 允许的源 (默认 ["*"]).
        """
        self._orch = orchestration_engine
        self._session = session_manager
        self._bus = message_bus
        self._agents = agent_runtime
        self._l1 = l1_gateway
        self._skills = skill_executor
        self._cors_origins = cors_origins or ["*"]
        self._handlers = _RouteHandlers(
            orchestration_engine=orchestration_engine,
            session_manager=session_manager,
            message_bus=message_bus,
            agent_runtime=agent_runtime,
            l1_gateway=l1_gateway,
            skill_executor=skill_executor,
        )

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例."""
        h = self._handlers

        routes = [
            # 健康检查
            Route("/health", h.health, methods=["GET"]),

            # Agent 运行时
            Route("/agents", h.agents, methods=["GET"]),
            Route("/agents/{agent_id}", h.agent_detail, methods=["GET"]),
            Route("/agents/{agent_id}/run", h.run_agent, methods=["POST"]),
            # 技能执行器
            Route("/agents/{agent_id}/skills", h.skills, methods=["GET"]),
            Route("/agents/{agent_id}/skills/call", h.call_skill, methods=["POST"]),
            # Agent 广播 inbox (按需协作)
            Route("/agents/{agent_id}/inbox", h.agent_inbox, methods=["GET"]),

            # 编排引擎
            Route("/orchestrate", h.orchestrate, methods=["POST"]),
            Route("/orchestration/{plan_id}", h.get_orchestration_result, methods=["GET"]),

            # 会话管理
            Route("/session", h.create_session, methods=["POST"]),
            Route("/session/{session_id}", h.get_session, methods=["GET"]),
            Route("/session/{session_id}/fork", h.fork_session, methods=["POST"]),
            Route("/session/{session_id}/close", h.close_session, methods=["POST"]),

            # 消息总线
            Route("/message/publish", h.publish_message, methods=["POST"]),

            # 交互记录
            Route("/interactions/stats", h.interaction_stats, methods=["GET"]),
            Route("/interactions/chains", h.interaction_chains, methods=["GET"]),
            Route("/interactions/chains/{chain_id}", h.interaction_chain_detail, methods=["GET"]),
            Route("/interactions/records", h.interaction_records, methods=["GET"]),
            Route("/interactions/latest", h.interaction_latest, methods=["GET"]),
            Route("/interactions/demo", h.interaction_demo, methods=["POST"]),
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
        """获取所有路由摘要 (用于文档/发现)."""
        return [
            {"path": "/health", "methods": ["GET"], "description": "L5 Agent Runtime 健康检查"},
            {"path": "/agents", "methods": ["GET"], "description": "列出已注册 Agent 与实例状态"},
            {"path": "/agents/{agent_id}", "methods": ["GET"], "description": "获取单个 Agent 状态"},
            {"path": "/agents/{agent_id}/run", "methods": ["POST"], "description": "执行单个 Agent worker"},
            {"path": "/orchestrate", "methods": ["POST"], "description": "执行编排计划 (Pipeline/Debate/Voting)"},
            {"path": "/orchestration/{plan_id}", "methods": ["GET"], "description": "获取编排结果"},
            {"path": "/session", "methods": ["POST"], "description": "创建会话"},
            {"path": "/session/{session_id}", "methods": ["GET"], "description": "获取会话状态"},
            {"path": "/session/{session_id}/fork", "methods": ["POST"], "description": "Fork 会话"},
            {"path": "/session/{session_id}/close", "methods": ["POST"], "description": "关闭会话"},
            {"path": "/message/publish", "methods": ["POST"], "description": "发布消息到总线"},
            {"path": "/interactions/stats", "methods": ["GET"], "description": "获取交互记录统计"},
            {"path": "/interactions/chains", "methods": ["GET"], "description": "获取交互链列表"},
            {"path": "/interactions/chains/{chain_id}", "methods": ["GET"], "description": "获取交互链详情"},
            {"path": "/interactions/records", "methods": ["GET"], "description": "获取交互记录列表"},
            {"path": "/interactions/latest", "methods": ["GET"], "description": "获取最新交互记录"},
            {"path": "/interactions/demo", "methods": ["POST"], "description": "生成演示交互数据"},
        ]


__all__ = ["L5Router"]
