"""CC2 计划审批门 — REST API 路由层.

将 CC2 计划-审批门控系统 (Plan-Approval Gate) 的全部子系统暴露为
RESTful JSON API, 提供 8 大端点组的完整 HTTP 接口。

基于 Starlette 构建 (与 L6 REST API 路由层保持一致的设计模式),
同时提供框架无关的纯方法接口, 支持直接编程调用。

设计原则:
- 统一响应格式: {"code": 200, "data": ..., "message": "OK"}
- 错误响应格式: {"code": <error_code>, "data": None, "message": <error_message>}
- 异常统一处理, CC2Error 自动映射为错误响应
- 线程安全: 所有共享状态通过 threading.RLock() 保护
- 子系统解耦: 路由引擎 / 审批工作流 / 干预管理 / 抗疲劳 / KPI 独立注入
- 可编程调用: 每个端点方法均可直接调用, 返回 dict

端点概览 (8 大组, 共 48 个端点):

    # 1. 路由决策 (Routing)
    POST   /cc2/route                      — 执行路由决策
    GET    /cc2/route/history               — 获取路由历史
    GET    /cc2/route/statistics            — 获取路由统计
    GET    /cc2/route/rules                 — 列出路由规则

    # 2. 审批工作流 (Approval)
    POST   /cc2/approval/create             — 创建审批请求
    POST   /cc2/approval/{request_id}/decision — 审批决策
    GET    /cc2/approval/{request_id}        — 获取审批记录
    GET    /cc2/approval/list                — 列出审批记录
    GET    /cc2/approval/statistics          — 审批统计
    POST   /cc2/approval/trust-mode/activate — 激活信任模式窗口
    POST   /cc2/approval/trust-mode/deactivate — 停用信任模式
    POST   /cc2/approval/rule-preset/add     — 添加规则预设
    DELETE /cc2/approval/rule-preset/{preset_id} — 移除规则预设

    # 3. 干预管理 (Intervention)
    POST   /cc2/intervention/emergency-pause            — 发起紧急暂停
    POST   /cc2/intervention/emergency-pause/{pause_id}/resolve — 解决紧急暂停
    POST   /cc2/intervention/manual-override            — 发起人工接管
    POST   /cc2/intervention/manual-override/{override_id}/release — 释放接管
    POST   /cc2/intervention/correction                 — 提交纠正反馈
    POST   /cc2/intervention/correction/{correction_id}/apply — 应用纠正
    POST   /cc2/intervention/creative-request           — 发起创意请求
    POST   /cc2/intervention/creative-request/{request_id}/respond — 响应创意请求
    GET    /cc2/intervention/active                     — 获取活跃干预
    GET    /cc2/intervention/history                    — 获取干预历史
    GET    /cc2/intervention/statistics                  — 干预统计

    # 4. 抗疲劳 (Anti-Fatigue)
    GET    /cc2/fatigue/{user_id}                        — 获取用户疲劳状态
    GET    /cc2/fatigue/{user_id}/adjustment             — 获取疲劳调整建议
    POST   /cc2/fatigue/batch/add                        — 添加批量审批项
    GET    /cc2/fatigue/batch/{user_id}/ready            — 获取就绪批量组
    POST   /cc2/fatigue/batch/{batch_id}/resolve         — 解决批量组
    GET    /cc2/fatigue/{user_id}/trust                  — 获取信任记录
    POST   /cc2/fatigue/{user_id}/trust/promote          — 提升信任
    POST   /cc2/fatigue/{user_id}/trust/demote           — 降低信任
    GET    /cc2/fatigue/statistics                       — 抗疲劳统计

    # 5. KPI 指标 (Metrics)
    GET    /cc2/kpi/dashboard                            — KPI 仪表盘
    GET    /cc2/kpi/{kpi_name}/summary                   — KPI 汇总
    GET    /cc2/kpi/{kpi_name}/history                   — KPI 历史
    GET    /cc2/kpi/alerts                               — 当前告警
    POST   /cc2/kpi/{kpi_name}/threshold                 — 调整 KPI 阈值

    # 6. 协作引擎 (Collaboration Engine)
    POST   /cc2/agent/register                           — 注册 Agent 配置
    GET    /cc2/agent/{agent_id}                         — 获取 Agent 配置
    PUT    /cc2/agent/{agent_id}                         — 更新 Agent 配置
    GET    /cc2/agent/list                               — 列出全部 Agent
    POST   /cc2/agent/{agent_id}/switch-mode             — 切换协作模式
    POST   /cc2/agent/{agent_id}/check-auto-step         — 检查自主步数
    GET    /cc2/agent/{agent_id}/interventions           — 获取 Agent 干预

    # 7. CC1 联动 (CC1 Integration)
    POST   /cc2/cc1/process-result                       — 处理 CC1 评审结果

    # 8. 健康检查 (Health)
    GET    /cc2/health                                   — 健康检查
    GET    /cc2/health/ready                             — 就绪检查 (含子系统状态)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .engine import CollaborationEngine
from .models import (
    AgentCollaborationProfile,
    CollaborationMode,
    HumanDecision,
    InterventionStatus,
    InterventionType,
    SwitchTrigger,
)
from .exceptions import CC2Error, ProfileNotFoundError
from .routing_engine import (
    RoutingContext,
    RoutingEngine,
    RoutingResult,
    RoutingRule,
    RiskLevel,
    Reversibility,
    UserRole,
    ApprovalMode,
    TimeoutAction,
    CollaborationLayer,
)
from .approval_workflow import (
    ApprovalStatus,
    ApprovalWorkflowManager,
)
from .intervention_manager import (
    CorrectionType,
    CorrectionSeverity,
    CreativeRequestType,
    InterventionManager,
    OverrideLevel,
    PauseScope,
    RecoveryMode,
)
from .anti_fatigue import AntiFatigueManager
from .kpi_metrics import KPIMetricsEngine, KPI_NAMES

logger = logging.getLogger("dy3_polaris.l0.cc2.api")

# Starlette 可选导入 — 未安装时仍可使用纯方法接口
try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    _STARLETTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STARLETTE_AVAILABLE = False
    Request = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    Route = None  # type: ignore[assignment,misc]
    Starlette = None  # type: ignore[assignment,misc]
    Middleware = None  # type: ignore[assignment,misc]
    CORSMiddleware = None  # type: ignore[assignment,misc]


# ============================================================
# 统一响应构造
# ============================================================


def _ok(data: Any = None, message: str = "OK") -> dict[str, Any]:
    """构造成功响应.

    Args:
        data: 响应数据
        message: 响应消息

    Returns:
        ``{"code": 200, "data": data, "message": message}``
    """
    return {"code": 200, "data": data, "message": message}


def _err(code: int, message: str) -> dict[str, Any]:
    """构造错误响应.

    Args:
        code: 错误码 (HTTP 风格)
        message: 错误消息

    Returns:
        ``{"code": code, "data": None, "message": message}``
    """
    return {"code": code, "data": None, "message": message}


def _serialize(obj: Any) -> Any:
    """递归序列化对象为 JSON 兼容格式.

    处理 Pydantic 模型、dataclass、枚举、列表和字典。
    """
    if obj is None:
        return None
    # Pydantic v2 模型
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    # 枚举
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    # dataclass (如 RoutingRule)
    if hasattr(obj, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
            # 跳过 callable 字段 (如 matcher 函数)
            if callable(field_value) and not hasattr(field_value, "model_dump"):
                result[field_name] = "<callable>"
            else:
                result[field_name] = _serialize(field_value)
        return result
    # 列表 / 元组
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    # 字典
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    # 基本类型
    return obj


# ============================================================
# CC2APIRouter
# ============================================================


class CC2APIRouter:
    """CC2 计划审批门 REST API 路由器.

    将 CC2 全部子系统 (路由引擎 / 审批工作流 / 干预管理 / 抗疲劳 /
    KPI 指标 / 协作引擎) 暴露为统一的 RESTful JSON API。

    设计模式与 L6 REST API 路由层 (``dy3_polaris.l6.api.router``) 保持一致,
    基于 Starlette 构建, 同时提供纯方法接口支持直接编程调用。

    统一响应格式:
        - 成功: ``{"code": 200, "data": <result>, "message": "OK"}``
        - 错误: ``{"code": <error_code>, "data": None, "message": <error_message>}``

    线程安全:
        所有共享状态通过 ``threading.RLock`` 保护,
        底层子系统亦各自维护独立锁。

    使用示例::

        # 1. 创建路由器
        engine = CollaborationEngine()
        router = CC2APIRouter(engine)

        # 2. 直接编程调用 (返回 dict)
        result = router.route({
            "operation_type": "learning_path_reset",
            "risk_level": "high",
            "confidence": 0.75,
            "reversibility": "irreversible",
        })
        print(result)  # {"code": 200, "data": {...}, "message": "OK"}

        # 3. 创建 Starlette 应用 (REST API)
        app = router.create_app()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # 4. 或挂载到现有 FastAPI 应用
        # from fastapi import FastAPI
        # fa = FastAPI()
        # fa.mount("/cc2", router.create_app())
    """

    def __init__(
        self,
        engine: CollaborationEngine,
        *,
        routing_engine: RoutingEngine | None = None,
        approval_manager: ApprovalWorkflowManager | None = None,
        intervention_manager: InterventionManager | None = None,
        anti_fatigue_manager: AntiFatigueManager | None = None,
        kpi_engine: KPIMetricsEngine | None = None,
    ) -> None:
        """初始化 CC2 API 路由器.

        Args:
            engine: 人机协作引擎实例 (核心, 必须提供)
            routing_engine: 六维决策路由引擎 (None 则自动创建)
            approval_manager: L3 审批工作流管理器 (None 则自动创建)
            intervention_manager: L4 干预层管理器 (None 则自动创建)
            anti_fatigue_manager: 审批抗疲劳机制管理器 (None 则自动创建)
            kpi_engine: KPI 指标引擎 (None 则自动创建)
        """
        self._engine = engine
        self._routing_engine = routing_engine or RoutingEngine()
        self._approval_manager = approval_manager or ApprovalWorkflowManager()
        self._intervention_manager = intervention_manager or InterventionManager()
        self._anti_fatigue_manager = anti_fatigue_manager or AntiFatigueManager()
        self._kpi_engine = kpi_engine or KPIMetricsEngine()
        self._lock = threading.RLock()
        self._started_at = time.time()

    # ==========================================================
    # 1. 路由决策 (Routing)
    # ==========================================================

    def route(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/route — 执行路由决策.

        接收六维路由上下文, 执行决策路由引擎的混合策略
        (规则覆盖 + 加权评分), 返回推荐协同层级和审批模式。

        请求体字段 (全部可选, 使用默认值):
            operation_type, target, risk_level, confidence,
            trust_score, reversibility, user_role, cognitive_load,
            user_id, session_id, metadata

        Returns:
            路由决策结果 RoutingResult (序列化为 dict)
        """
        try:
            ctx = RoutingContext(**body)
        except Exception as exc:
            return _err(400, f"路由上下文构造失败: {exc}")

        try:
            result = self._routing_engine.route(ctx)
            return _ok(_serialize(result))
        except CC2Error as exc:
            return _err(500, f"[{exc.code}] {exc.detail}")
        except Exception as exc:
            return _err(500, f"路由决策执行失败: {exc}")

    def get_route_history(self, limit: int = 100) -> dict[str, Any]:
        """GET /cc2/route/history — 获取路由历史.

        Args:
            limit: 返回最大数量 (默认 100)

        Returns:
            路由决策结果列表
        """
        history = self._routing_engine.routing_history
        return _ok(_serialize(history[-limit:]))

    def get_route_statistics(self) -> dict[str, Any]:
        """GET /cc2/route/statistics — 获取路由统计.

        Returns:
            按层级、规则分组的路由统计信息
        """
        stats = self._routing_engine.get_statistics()
        return _ok(_serialize(stats))

    def get_route_rules(self) -> dict[str, Any]:
        """GET /cc2/route/rules — 列出路由规则.

        Returns:
            全部场景化路由规则列表 (按优先级排序)
        """
        rules = self._routing_engine.rules
        return _ok(_serialize(rules))

    # ==========================================================
    # 2. 审批工作流 (Approval)
    # ==========================================================

    def create_approval(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/approval/create — 创建审批请求.

        请求体字段:
            operation (必填), target, risk_level, reversibility,
            approval_mode, requester, approver_roles, timeout_seconds,
            timeout_action, alternatives, context, policy_reference, user_id

        如果用户有有效信任模式窗口且操作非安全相关, 自动批准。

        Returns:
            审批请求 ApprovalRequest (序列化为 dict)
        """
        try:
            request = self._approval_manager.create_request(
                operation=body.get("operation", ""),
                target=body.get("target", ""),
                risk_level=RiskLevel(body.get("risk_level", "medium")),
                reversibility=Reversibility(
                    body.get("reversibility", "partially_reversible")
                ),
                approval_mode=ApprovalMode(
                    body.get("approval_mode", "detailed_review")
                ),
                requester=body.get("requester", ""),
                approver_roles=[
                    UserRole(r) if isinstance(r, str) else r
                    for r in body.get("approver_roles", ["student"])
                ],
                timeout_seconds=body.get("timeout_seconds", 300.0),
                timeout_action=TimeoutAction(
                    body.get("timeout_action", "abort")
                ),
                alternatives=body.get("alternatives", []),
                context=body.get("context", {}),
                policy_reference=body.get("policy_reference", ""),
                user_id=body.get("user_id", ""),
            )
            return _ok(_serialize(request))
        except CC2Error as exc:
            return _err(500, f"[{exc.code}] {exc.detail}")
        except Exception as exc:
            return _err(500, f"审批请求创建失败: {exc}")

    def make_approval_decision(
        self, request_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/approval/{request_id}/decision — 审批决策.

        请求体字段:
            decision (approved/rejected/modified), decided_by,
            comment, selected_alternative, modified_parameters

        Returns:
            更新后的审批记录 ApprovalRecord
        """
        try:
            record = self._approval_manager.make_decision(
                request_id=request_id,
                decision=ApprovalStatus(body.get("decision", "approved")),
                decided_by=body.get("decided_by", ""),
                comment=body.get("comment", ""),
                selected_alternative=body.get("selected_alternative", ""),
                modified_parameters=body.get("modified_parameters", {}),
            )
            return _ok(_serialize(record))
        except KeyError as exc:
            return _err(404, str(exc))
        except ValueError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"审批决策失败: {exc}")

    def get_approval(self, request_id: str) -> dict[str, Any]:
        """GET /cc2/approval/{request_id} — 获取审批记录.

        Args:
            request_id: 审批请求 ID

        Returns:
            审批记录 ApprovalRecord
        """
        record = self._approval_manager.get_record(request_id)
        if record is None:
            return _err(404, f"审批记录不存在: {request_id}")
        return _ok(_serialize(record))

    def list_approvals(
        self,
        status: str | None = None,
        operation: str | None = None,
        requester: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """GET /cc2/approval/list — 列出审批记录.

        支持按状态、操作类型、请求方过滤。

        Args:
            status: 审批状态过滤
            operation: 操作类型过滤
            requester: 请求方过滤
            limit: 返回最大数量

        Returns:
            审批记录列表
        """
        status_enum = None
        if status:
            try:
                status_enum = ApprovalStatus(status)
            except ValueError:
                return _err(400, f"无效的审批状态: {status}")

        records = self._approval_manager.list_records(
            status=status_enum,
            operation=operation,
            requester=requester,
            limit=limit,
        )
        return _ok(_serialize(records))

    def get_approval_statistics(self) -> dict[str, Any]:
        """GET /cc2/approval/statistics — 审批统计.

        Returns:
            按状态、模式分组的审批统计信息
        """
        stats = self._approval_manager.get_statistics()
        return _ok(_serialize(stats))

    def activate_trust_mode(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/approval/trust-mode/activate — 激活信任模式窗口.

        请求体字段:
            user_id (必填), duration_seconds (默认 1800=30 分钟)

        信任模式窗口内, 低风险可逆操作自动批准。

        Returns:
            信任模式窗口信息
        """
        user_id = body.get("user_id", "")
        if not user_id:
            return _err(400, "缺少 user_id 字段")
        duration = body.get("duration_seconds", 1800.0)
        try:
            window = self._approval_manager.activate_trust_mode(
                user_id=user_id,
                duration_seconds=duration,
            )
            return _ok({
                "user_id": window.user_id,
                "duration_seconds": window.duration_seconds,
                "start_time": window.start_time,
                "active": window.active,
                "remaining_seconds": window.remaining_seconds(),
            })
        except Exception as exc:
            return _err(500, f"信任模式激活失败: {exc}")

    def deactivate_trust_mode(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/approval/trust-mode/deactivate — 停用信任模式.

        请求体字段:
            user_id (必填)

        Returns:
            操作结果
        """
        user_id = body.get("user_id", "")
        if not user_id:
            return _err(400, "缺少 user_id 字段")
        self._approval_manager.deactivate_trust_mode(user_id)
        return _ok({"user_id": user_id, "deactivated": True})

    def add_rule_preset(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/approval/rule-preset/add — 添加规则预设.

        请求体字段:
            preset_id (必填), operation (必填), risk_level, action

        规则预设允许用户批量设置审批规则, 如 "总是批准此类操作"。

        Returns:
            规则预设信息
        """
        preset_id = body.get("preset_id", "")
        operation = body.get("operation", "")
        if not preset_id or not operation:
            return _err(400, "缺少 preset_id 或 operation 字段")
        risk_level = RiskLevel(body.get("risk_level", "low"))
        action = body.get("action", "auto_approve")
        self._approval_manager.add_rule_preset(
            preset_id=preset_id,
            operation=operation,
            risk_level=risk_level,
            action=action,
        )
        return _ok({
            "preset_id": preset_id,
            "operation": operation,
            "risk_level": risk_level.value,
            "action": action,
        })

    def remove_rule_preset(self, preset_id: str) -> dict[str, Any]:
        """DELETE /cc2/approval/rule-preset/{preset_id} — 移除规则预设.

        Args:
            preset_id: 规则预设 ID

        Returns:
            操作结果
        """
        self._approval_manager.remove_rule_preset(preset_id)
        return _ok({"preset_id": preset_id, "removed": True})

    # ==========================================================
    # 3. 干预管理 (Intervention)
    # ==========================================================

    def initiate_emergency_pause(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/intervention/emergency-pause — 发起紧急暂停.

        立即创建紧急暂停请求, 阻塞全部相关 Agent 执行。
        紧急暂停是最高优先级干预 (P0), 响应时间 < 5 秒。

        请求体字段:
            user_id (必填), reason, scope, agent_ids,
            auto_notify_teacher, risk_level, recovery_mode

        Returns:
            紧急暂停请求 EmergencyPauseRequest
        """
        user_id = body.get("user_id", "")
        if not user_id:
            return _err(400, "缺少 user_id 字段")
        try:
            pause = self._intervention_manager.initiate_emergency_pause(
                user_id=user_id,
                reason=body.get("reason", ""),
                scope=PauseScope(body.get("scope", "session")),
                agent_ids=body.get("agent_ids", []),
                auto_notify_teacher=body.get("auto_notify_teacher", True),
                risk_level=RiskLevel(body.get("risk_level", "high")),
                recovery_mode=RecoveryMode(
                    body.get("recovery_mode", "resume_from_checkpoint")
                ),
            )
            return _ok(_serialize(pause))
        except Exception as exc:
            return _err(500, f"紧急暂停发起失败: {exc}")

    def resolve_emergency_pause(
        self, pause_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/intervention/emergency-pause/{pause_id}/resolve — 解决紧急暂停.

        请求体字段:
            resolution (必填), resolved_by (必填)

        Returns:
            已解决的紧急暂停请求
        """
        try:
            pause = self._intervention_manager.resolve_emergency_pause(
                pause_id=pause_id,
                resolution=body.get("resolution", ""),
                resolved_by=body.get("resolved_by", ""),
            )
            return _ok(_serialize(pause))
        except KeyError as exc:
            return _err(404, str(exc))
        except ValueError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"紧急暂停解决失败: {exc}")

    def initiate_manual_override(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/intervention/manual-override — 发起人工接管.

        人类操作者从 AI Agent 接管控制权。

        请求体字段:
            operator_id (必填), target_agent (必填), override_level,
            instructions, duration_seconds, context, operator_role,
            risk_level, recovery_mode

        Returns:
            人工接管请求 ManualOverrideRequest
        """
        operator_id = body.get("operator_id", "")
        target_agent = body.get("target_agent", "")
        if not operator_id or not target_agent:
            return _err(400, "缺少 operator_id 或 target_agent 字段")
        try:
            override = self._intervention_manager.initiate_manual_override(
                operator_id=operator_id,
                target_agent=target_agent,
                override_level=OverrideLevel(
                    body.get("override_level", "executive")
                ),
                instructions=body.get("instructions", ""),
                duration_seconds=body.get("duration_seconds"),
                context=body.get("context", {}),
                operator_role=UserRole(body.get("operator_role", "teacher")),
                risk_level=RiskLevel(body.get("risk_level", "medium")),
                recovery_mode=RecoveryMode(
                    body.get("recovery_mode", "resume_from_checkpoint")
                ),
            )
            return _ok(_serialize(override))
        except Exception as exc:
            return _err(500, f"人工接管发起失败: {exc}")

    def release_override(
        self, override_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/intervention/manual-override/{override_id}/release — 释放接管.

        请求体字段:
            summary, released_by

        Returns:
            已释放的人工接管请求
        """
        try:
            override = self._intervention_manager.release_override(
                override_id=override_id,
                summary=body.get("summary", ""),
                released_by=body.get("released_by", ""),
            )
            return _ok(_serialize(override))
        except KeyError as exc:
            return _err(404, str(exc))
        except ValueError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"人工接管释放失败: {exc}")

    def submit_correction(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/intervention/correction — 提交纠正反馈.

        人类纠正 AI 输出并提供反馈, 纠正结果可触发 CC1 重新评审。

        请求体字段:
            corrector_id (必填), target_content_id (必填),
            original, corrected, correction_type (必填),
            feedback, severity, target_agent_id,
            corrector_role, learning_value, metadata

        Returns:
            纠正反馈对象 CorrectionFeedback
        """
        corrector_id = body.get("corrector_id", "")
        target_content_id = body.get("target_content_id", "")
        correction_type_str = body.get("correction_type", "")
        if not corrector_id or not target_content_id:
            return _err(400, "缺少 corrector_id 或 target_content_id 字段")
        if not correction_type_str:
            return _err(400, "缺少 correction_type 字段")
        try:
            correction = self._intervention_manager.submit_correction(
                corrector_id=corrector_id,
                target_content_id=target_content_id,
                original=body.get("original", ""),
                corrected=body.get("corrected", ""),
                correction_type=CorrectionType(correction_type_str),
                feedback=body.get("feedback", ""),
                severity=CorrectionSeverity(
                    body.get("severity", "moderate")
                ),
                target_agent_id=body.get("target_agent_id", ""),
                corrector_role=UserRole(body.get("corrector_role", "teacher")),
                learning_value=body.get("learning_value", 0.5),
                metadata=body.get("metadata", {}),
            )
            return _ok(_serialize(correction))
        except Exception as exc:
            return _err(500, f"纠正反馈提交失败: {exc}")

    def apply_correction(
        self, correction_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/intervention/correction/{correction_id}/apply — 应用纠正.

        请求体字段:
            applied_by (必填), trigger_cc1, cc1_review_id

        Returns:
            已应用的纠正反馈对象
        """
        applied_by = body.get("applied_by", "")
        if not applied_by:
            return _err(400, "缺少 applied_by 字段")
        try:
            correction = self._intervention_manager.apply_correction(
                correction_id=correction_id,
                applied_by=applied_by,
                trigger_cc1=body.get("trigger_cc1"),
                cc1_review_id=body.get("cc1_review_id", ""),
            )
            return _ok(_serialize(correction))
        except KeyError as exc:
            return _err(404, str(exc))
        except ValueError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"纠正应用失败: {exc}")

    def request_creative_input(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/intervention/creative-request — 发起创意请求.

        人类请求 AI 提供创意/发散性输入, 作为人机共创的协商回合。

        请求体字段:
            requester_id (必填), request_type (必填),
            topic, constraints, desired_output_format,
            context, requester_role, metadata

        Returns:
            创意请求对象 CreativeRequest
        """
        requester_id = body.get("requester_id", "")
        request_type_str = body.get("request_type", "")
        if not requester_id:
            return _err(400, "缺少 requester_id 字段")
        if not request_type_str:
            return _err(400, "缺少 request_type 字段")
        try:
            request = self._intervention_manager.request_creative_input(
                requester_id=requester_id,
                request_type=CreativeRequestType(request_type_str),
                topic=body.get("topic", ""),
                constraints=body.get("constraints", []),
                desired_output_format=body.get("desired_output_format", ""),
                context=body.get("context", {}),
                requester_role=UserRole(body.get("requester_role", "teacher")),
                metadata=body.get("metadata", {}),
            )
            return _ok(_serialize(request))
        except Exception as exc:
            return _err(500, f"创意请求发起失败: {exc}")

    def respond_creative_request(
        self, request_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/intervention/creative-request/{request_id}/respond — 响应创意请求.

        请求体字段:
            responder_id (必填), content (必填)

        Returns:
            已响应的创意请求对象
        """
        responder_id = body.get("responder_id", "")
        content = body.get("content", "")
        if not responder_id:
            return _err(400, "缺少 responder_id 字段")
        if not content:
            return _err(400, "缺少 content 字段")
        try:
            request = self._intervention_manager.respond_creative_request(
                request_id=request_id,
                responder_id=responder_id,
                content=content,
            )
            return _ok(_serialize(request))
        except KeyError as exc:
            return _err(404, str(exc))
        except ValueError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"创意请求响应失败: {exc}")

    def get_active_interventions(self) -> dict[str, Any]:
        """GET /cc2/intervention/active — 获取活跃干预.

        返回所有状态为 INITIATED 或 ACTIVE 的干预事件。

        Returns:
            活跃干预事件列表
        """
        events = self._intervention_manager.get_active_interventions()
        return _ok(_serialize(events))

    def get_intervention_history(
        self,
        limit: int = 100,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """GET /cc2/intervention/history — 获取干预历史.

        Args:
            limit: 返回最大数量
            event_type: 干预类型过滤 (emergency_pause/manual_override/
                        correction_feedback/creative_request)

        Returns:
            干预事件列表 (按创建时间降序)
        """
        from .routing_engine import InterventionTypeL4

        type_enum = None
        if event_type:
            try:
                type_enum = InterventionTypeL4(event_type)
            except ValueError:
                return _err(400, f"无效的干预类型: {event_type}")

        events = self._intervention_manager.get_intervention_history(
            limit=limit,
            event_type=type_enum,
        )
        return _ok(_serialize(events))

    def get_intervention_statistics(self) -> dict[str, Any]:
        """GET /cc2/intervention/statistics — 干预统计.

        Returns:
            按类型、状态分组的干预统计信息
        """
        stats = self._intervention_manager.get_statistics()
        return _ok(_serialize(stats))

    # ==========================================================
    # 4. 抗疲劳 (Anti-Fatigue)
    # ==========================================================

    def get_fatigue_state(self, user_id: str) -> dict[str, Any]:
        """GET /cc2/fatigue/{user_id} — 获取用户疲劳状态.

        Args:
            user_id: 用户 ID

        Returns:
            用户疲劳状态 FatigueState
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        state = self._anti_fatigue_manager.get_fatigue_state(user_id)
        return _ok(_serialize(state))

    def get_fatigue_adjustment(self, user_id: str) -> dict[str, Any]:
        """GET /cc2/fatigue/{user_id}/adjustment — 获取疲劳调整建议.

        根据用户疲劳等级返回调整建议:
        - NONE: 无调整
        - MILD: 建议启用批量审批
        - MODERATE: 建议启用智能预批
        - SEVERE: 建议降级到 L2 提示层

        Args:
            user_id: 用户 ID

        Returns:
            疲劳调整建议字典
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        adjustment = self._anti_fatigue_manager.get_fatigue_adjustment(user_id)
        return _ok(_serialize(adjustment))

    def add_to_batch(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/fatigue/batch/add — 添加批量审批项.

        将审批请求添加到批量组, 相同操作类型的请求在批量窗口内聚合,
        达到最小批量大小后返回就绪的批量组。

        请求体字段:
            user_id (必填), operation (必填), request_data (必填)

        Returns:
            就绪的批量组 (如果达到最小批量大小), 否则 None
        """
        user_id = body.get("user_id", "")
        operation = body.get("operation", "")
        request_data = body.get("request_data", {})
        if not user_id or not operation:
            return _err(400, "缺少 user_id 或 operation 字段")
        batch = self._anti_fatigue_manager.add_to_batch(
            user_id=user_id,
            operation=operation,
            request_data=request_data,
        )
        if batch is None:
            return _ok({"batch_ready": False, "message": "批量组收集中, 尚未达到最小批量大小"})
        return _ok(_serialize(batch))

    def get_ready_batches(self, user_id: str) -> dict[str, Any]:
        """GET /cc2/fatigue/batch/{user_id}/ready — 获取就绪批量组.

        Args:
            user_id: 用户 ID

        Returns:
            就绪的批量审批组列表
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        batches = self._anti_fatigue_manager.get_ready_batches(user_id)
        return _ok(_serialize(batches))

    def resolve_batch(self, batch_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/fatigue/batch/{batch_id}/resolve — 解决批量组.

        请求体字段:
            decision (approved/rejected, 必填), decided_by

        Returns:
            更新后的批量组
        """
        decision = body.get("decision", "")
        if not decision:
            return _err(400, "缺少 decision 字段")
        batch = self._anti_fatigue_manager.resolve_batch(
            batch_id=batch_id,
            decision=decision,
            decided_by=body.get("decided_by", ""),
        )
        if batch is None:
            return _err(404, f"批量组不存在或状态不允许解决: {batch_id}")
        return _ok(_serialize(batch))

    def get_trust_record(self, user_id: str) -> dict[str, Any]:
        """GET /cc2/fatigue/{user_id}/trust — 获取信任记录.

        Args:
            user_id: 用户 ID

        Returns:
            渐进信任记录 ProgressiveTrustRecord
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        trust = self._anti_fatigue_manager.get_trust_record(user_id)
        return _ok(_serialize(trust))

    def promote_trust(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/fatigue/{user_id}/trust/promote — 提升信任.

        请求体字段:
            reason (可选)

        Returns:
            提升后的信任度
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        new_score = self._anti_fatigue_manager.promote_trust(
            user_id=user_id,
            reason=body.get("reason", ""),
        )
        return _ok({"user_id": user_id, "trust_score": new_score})

    def demote_trust(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/fatigue/{user_id}/trust/demote — 降低信任.

        请求体字段:
            reason (可选)

        Returns:
            降低后的信任度
        """
        if not user_id:
            return _err(400, "缺少 user_id")
        new_score = self._anti_fatigue_manager.demote_trust(
            user_id=user_id,
            reason=body.get("reason", ""),
        )
        return _ok({"user_id": user_id, "trust_score": new_score})

    def get_fatigue_statistics(self) -> dict[str, Any]:
        """GET /cc2/fatigue/statistics — 抗疲劳统计.

        Returns:
            全局抗疲劳统计信息
        """
        stats = self._anti_fatigue_manager.get_statistics()
        return _ok(_serialize(stats))

    # ==========================================================
    # 5. KPI 指标 (Metrics)
    # ==========================================================

    def get_kpi_dashboard(self) -> dict[str, Any]:
        """GET /cc2/kpi/dashboard — KPI 仪表盘.

        生成包含所有 KPI 状态、整体健康分、分类得分、告警与统计信息的
        仪表盘数据包。

        Returns:
            仪表盘数据字典
        """
        data = self._kpi_engine.get_dashboard_data()
        return _ok(_serialize(data))

    def get_kpi_summary(self, kpi_name: str) -> dict[str, Any]:
        """GET /cc2/kpi/{kpi_name}/summary — KPI 汇总.

        Args:
            kpi_name: KPI 标识名称

        Returns:
            KPI 汇总信息 KPISummary
        """
        try:
            summary = self._kpi_engine.get_kpi_summary(kpi_name)
            return _ok(_serialize(summary))
        except KeyError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"KPI 汇总获取失败: {exc}")

    def get_kpi_history(self, kpi_name: str, limit: int = 100) -> dict[str, Any]:
        """GET /cc2/kpi/{kpi_name}/history — KPI 历史.

        Args:
            kpi_name: KPI 标识名称
            limit: 返回最大记录数

        Returns:
            KPI 采样记录列表
        """
        try:
            history = self._kpi_engine.get_kpi_history(kpi_name, limit=limit)
            return _ok(_serialize(history))
        except KeyError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"KPI 历史获取失败: {exc}")

    def get_kpi_alerts(self) -> dict[str, Any]:
        """GET /cc2/kpi/alerts — 当前告警.

        返回所有当前处于黄色或红色状态的 KPI 告警。

        Returns:
            告警字典列表
        """
        alerts = self._kpi_engine.get_alerts()
        return _ok(_serialize(alerts))

    def adjust_kpi_threshold(
        self, kpi_name: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/kpi/{kpi_name}/threshold — 调整 KPI 阈值.

        请求体字段:
            green_max (可选, None 则自动计算), yellow_max (可选)

        Returns:
            更新后的阈值配置
        """
        try:
            threshold = self._kpi_engine.adjust_threshold(
                kpi_name=kpi_name,
                green_max=body.get("green_max"),
                yellow_max=body.get("yellow_max"),
            )
            return _ok(_serialize(threshold))
        except KeyError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"KPI 阈值调整失败: {exc}")

    # ==========================================================
    # 6. 协作引擎 (Collaboration Engine)
    # ==========================================================

    def register_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/agent/register — 注册 Agent 配置.

        请求体字段:
            agent_id (必填), mode, default_mode, max_auto_steps,
            confidence_threshold, timeout_seconds, escalation_targets,
            enabled, tags

        Returns:
            注册的 Agent 协作配置
        """
        agent_id = body.get("agent_id", "")
        if not agent_id:
            return _err(400, "缺少 agent_id 字段")
        try:
            profile = AgentCollaborationProfile(
                agent_id=agent_id,
                mode=CollaborationMode(body.get("mode", "conditional")),
                default_mode=CollaborationMode(
                    body.get("default_mode", "conditional")
                ),
                max_auto_steps=body.get("max_auto_steps", 10),
                confidence_threshold=body.get("confidence_threshold", 0.7),
                timeout_seconds=body.get("timeout_seconds", 300.0),
                escalation_targets=body.get("escalation_targets", []),
                enabled=body.get("enabled", True),
                tags=body.get("tags", []),
            )
            self._engine.register_profile(profile)
            return _ok(_serialize(profile))
        except Exception as exc:
            return _err(500, f"Agent 配置注册失败: {exc}")

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """GET /cc2/agent/{agent_id} — 获取 Agent 配置.

        Args:
            agent_id: Agent ID

        Returns:
            Agent 协作配置 AgentCollaborationProfile
        """
        try:
            profile = self._engine.get_profile(agent_id)
            return _ok(_serialize(profile))
        except ProfileNotFoundError as exc:
            return _err(404, f"[{exc.code}] {exc.detail}")
        except Exception as exc:
            return _err(500, f"Agent 配置获取失败: {exc}")

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """PUT /cc2/agent/{agent_id} — 更新 Agent 配置.

        请求体字段: 任意可更新的 AgentCollaborationProfile 字段。

        Returns:
            更新后的 Agent 协作配置
        """
        try:
            # 处理枚举字段
            kwargs: dict[str, Any] = {}
            for key, value in body.items():
                if key in ("mode", "default_mode") and isinstance(value, str):
                    kwargs[key] = CollaborationMode(value)
                else:
                    kwargs[key] = value
            profile = self._engine.update_profile(agent_id, **kwargs)
            return _ok(_serialize(profile))
        except ProfileNotFoundError as exc:
            return _err(404, f"[{exc.code}] {exc.detail}")
        except Exception as exc:
            return _err(500, f"Agent 配置更新失败: {exc}")

    def list_agents(self) -> dict[str, Any]:
        """GET /cc2/agent/list — 列出全部 Agent.

        Returns:
            全部已注册的 Agent 协作配置列表
        """
        profiles = self._engine.list_profiles()
        return _ok(_serialize(profiles))

    def switch_agent_mode(
        self, agent_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/agent/{agent_id}/switch-mode — 切换协作模式.

        请求体字段:
            to_mode (必填), trigger (必填), reason,
            confidence, allow_skip

        默认仅允许相邻级切换 (AutoGen 渐进自主启发),
        设置 allow_skip=True 可跳级 (如混沌感知紧急升级)。

        Returns:
            模式切换事件 ModeSwitchEvent
        """
        to_mode_str = body.get("to_mode", "")
        trigger_str = body.get("trigger", "")
        if not to_mode_str or not trigger_str:
            return _err(400, "缺少 to_mode 或 trigger 字段")
        try:
            event = self._engine.switch_mode(
                agent_id=agent_id,
                to_mode=CollaborationMode(to_mode_str),
                trigger=SwitchTrigger(trigger_str),
                reason=body.get("reason", ""),
                confidence=body.get("confidence"),
                allow_skip=body.get("allow_skip", False),
            )
            return _ok(_serialize(event))
        except ProfileNotFoundError as exc:
            return _err(404, f"[{exc.code}] {exc.detail}")
        except CC2Error as exc:
            return _err(409, f"[{exc.code}] {exc.detail}")
        except Exception as exc:
            return _err(500, f"模式切换失败: {exc}")

    def check_auto_step(
        self, agent_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc2/agent/{agent_id}/check-auto-step — 检查自主步数.

        在每步 Agent 执行前调用, 检查是否需要人类干预:
        1. SUPERVISED 模式: 每步都需要审批
        2. 置信度低于阈值: 创建升级干预
        3. 连续自主步数达上限: 创建检查点干预

        请求体字段:
            confidence (可选, 默认 1.0)

        Returns:
            干预记录 (如果需要干预), 否则 None
        """
        try:
            confidence = body.get("confidence", 1.0)
            record = self._engine.check_auto_step(
                agent_id=agent_id,
                confidence=confidence,
            )
            if record is None:
                return _ok(None, "无需干预, 继续自主执行")
            return _ok(_serialize(record))
        except Exception as exc:
            return _err(500, f"自主步数检查失败: {exc}")

    def get_agent_interventions(
        self, agent_id: str, limit: int = 100
    ) -> dict[str, Any]:
        """GET /cc2/agent/{agent_id}/interventions — 获取 Agent 干预.

        Args:
            agent_id: Agent ID
            limit: 返回最大数量

        Returns:
            该 Agent 的干预记录列表
        """
        records = self._engine.query_interventions(
            agent_id=agent_id,
            limit=limit,
        )
        return _ok(_serialize(records))

    # ==========================================================
    # 7. CC1 联动 (CC1 Integration)
    # ==========================================================

    def process_cc1_result(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc2/cc1/process-result — 处理 CC1 评审结果.

        接收 CC1 四层评审结果, 执行联动处理:
        1. 将 CC1 评审结果作为路由上下文, 执行决策路由
        2. 如果 CC1 verdict 为 "block", 自动创建 L3 审批请求 (详细审批)
        3. 记录 KPI 指标 (cc1_integration_rate)

        请求体字段:
            verdict (pass/block/warn, 必填),
            confidence (0-1, 必填),
            operation_type, target, risk_level, user_id,
            session_id, review_id, details

        Returns:
            联动处理结果 (路由结果 + 审批请求 + KPI 记录)
        """
        verdict = body.get("verdict", "")
        confidence = body.get("confidence")
        if not verdict:
            return _err(400, "缺少 verdict 字段")
        if confidence is None:
            return _err(400, "缺少 confidence 字段")

        try:
            result: dict[str, Any] = {
                "cc1_verdict": verdict,
                "cc1_confidence": confidence,
                "review_id": body.get("review_id", ""),
            }

            # 1. 构造路由上下文并执行路由
            routing_ctx_body: dict[str, Any] = {
                "operation_type": body.get("operation_type", "cc1_review"),
                "target": body.get("target", ""),
                "confidence": confidence,
                "risk_level": body.get("risk_level", "medium" if verdict != "block" else "high"),
                "user_id": body.get("user_id", ""),
                "session_id": body.get("session_id", ""),
                "metadata": {
                    "cc1_verdict": verdict,
                    "cc1_review_id": body.get("review_id", ""),
                    **body.get("details", {}),
                },
            }
            try:
                ctx = RoutingContext(**routing_ctx_body)
                routing_result = self._routing_engine.route(ctx)
                result["routing"] = _serialize(routing_result)
            except Exception as exc:
                result["routing_error"] = str(exc)

            # 2. CC1 block → 自动创建 L3 审批请求
            approval_created = False
            if verdict == "block":
                try:
                    approval_request = self._approval_manager.create_request(
                        operation=body.get("operation_type", "cc1_block_arbitration"),
                        target=body.get("target", ""),
                        risk_level=RiskLevel(
                            body.get("risk_level", "high")
                        ),
                        reversibility=Reversibility(
                            body.get("reversibility", "partially_reversible")
                        ),
                        approval_mode=ApprovalMode.DETAILED_REVIEW,
                        requester=body.get("user_id", "cc1"),
                        timeout_seconds=300.0,
                        timeout_action=TimeoutAction.ABORT,
                        context={
                            "cc1_verdict": verdict,
                            "cc1_review_id": body.get("review_id", ""),
                            "cc1_confidence": confidence,
                        },
                        policy_reference="RR-005",
                        user_id=body.get("user_id", ""),
                    )
                    result["approval_request"] = _serialize(approval_request)
                    approval_created = True
                except Exception as exc:
                    result["approval_error"] = str(exc)
            result["approval_created"] = approval_created

            # 3. 记录 KPI 指标
            try:
                self._kpi_engine.ingest_from_routing_engine(
                    cc1_triggered=(verdict in ("block", "warn")),
                    context={
                        "session_id": body.get("session_id", ""),
                        "review_id": body.get("review_id", ""),
                    },
                )
                result["kpi_recorded"] = True
            except Exception as exc:
                result["kpi_error"] = str(exc)

            return _ok(result)
        except Exception as exc:
            return _err(500, f"CC1 评审结果处理失败: {exc}")

    # ==========================================================
    # 8. 健康检查 (Health)
    # ==========================================================

    def health(self) -> dict[str, Any]:
        """GET /cc2/health — 健康检查.

        Returns:
            健康状态字典
        """
        return _ok({
            "status": "healthy",
            "service": "cc2-plan-approval-gate",
            "uptime_seconds": round(time.time() - self._started_at, 3),
        })

    def health_ready(self) -> dict[str, Any]:
        """GET /cc2/health/ready — 就绪检查 (含子系统状态).

        检查全部子系统的就绪状态:
        - collaboration_engine: 协作引擎
        - routing_engine: 路由引擎
        - approval_manager: 审批工作流管理器
        - intervention_manager: 干预管理器
        - anti_fatigue_manager: 抗疲劳管理器
        - kpi_engine: KPI 指标引擎

        Returns:
            就绪状态及各子系统状态
        """
        subsystems: dict[str, dict[str, Any]] = {}
        all_ready = True

        # 协作引擎
        try:
            stats = self._engine.get_stats()
            subsystems["collaboration_engine"] = {
                "ready": True,
                "registered_agents": stats.get("registered_agents", 0),
            }
        except Exception as exc:
            subsystems["collaboration_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 路由引擎
        try:
            route_stats = self._routing_engine.get_statistics()
            subsystems["routing_engine"] = {
                "ready": True,
                "total_routes": route_stats.get("total", 0),
                "rules_count": len(self._routing_engine.rules),
            }
        except Exception as exc:
            subsystems["routing_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 审批工作流
        try:
            approval_stats = self._approval_manager.get_statistics()
            subsystems["approval_manager"] = {
                "ready": True,
                "total_records": approval_stats.get("total", 0),
            }
        except Exception as exc:
            subsystems["approval_manager"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 干预管理器
        try:
            intv_stats = self._intervention_manager.get_statistics()
            subsystems["intervention_manager"] = {
                "ready": True,
                "total_events": intv_stats.get("total_events", 0),
                "active_count": intv_stats.get("active_count", 0),
            }
        except Exception as exc:
            subsystems["intervention_manager"] = {
                "ready": False,
                "error": str(exc),
            }
            all_ready = False

        # 抗疲劳管理器
        try:
            fatigue_stats = self._anti_fatigue_manager.get_statistics()
            subsystems["anti_fatigue_manager"] = {
                "ready": True,
                "total_users": fatigue_stats.get("total_users", 0),
            }
        except Exception as exc:
            subsystems["anti_fatigue_manager"] = {
                "ready": False,
                "error": str(exc),
            }
            all_ready = False

        # KPI 引擎
        try:
            kpi_stats = self._kpi_engine.get_statistics()
            subsystems["kpi_engine"] = {
                "ready": True,
                "tracked_kpis": kpi_stats.get("tracked_kpis", 0),
                "total_samples": kpi_stats.get("total_samples", 0),
            }
        except Exception as exc:
            subsystems["kpi_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        return _ok({
            "ready": all_ready,
            "status": "ready" if all_ready else "degraded",
            "subsystems": subsystems,
        })

    # ==========================================================
    # 路由定义与 Starlette 集成
    # ==========================================================

    def get_routes(self) -> list[dict[str, Any]]:
        """获取全部路由定义.

        返回所有 REST 端点的元数据, 用于路由注册、文档生成和 API 发现。

        每条路由定义包含:
            - path: URL 路径
            - methods: HTTP 方法列表
            - handler: CC2APIRouter 方法名
            - description: 端点描述
            - body: 是否需要请求体 (POST/PUT/DELETE)
            - path_params: 路径参数名列表
            - query_params: 查询参数名列表

        Returns:
            路由定义字典列表
        """
        return [
            # 1. 路由决策
            {
                "path": "/cc2/route",
                "methods": ["POST"],
                "handler": "route",
                "description": "执行路由决策 (六维输入 → 四层协同)",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/route/history",
                "methods": ["GET"],
                "handler": "get_route_history",
                "description": "获取路由历史",
                "body": False,
                "path_params": [],
                "query_params": ["limit"],
            },
            {
                "path": "/cc2/route/statistics",
                "methods": ["GET"],
                "handler": "get_route_statistics",
                "description": "获取路由统计",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/route/rules",
                "methods": ["GET"],
                "handler": "get_route_rules",
                "description": "列出路由规则",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            # 2. 审批工作流
            {
                "path": "/cc2/approval/create",
                "methods": ["POST"],
                "handler": "create_approval",
                "description": "创建审批请求",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/{request_id}/decision",
                "methods": ["POST"],
                "handler": "make_approval_decision",
                "description": "审批决策",
                "body": True,
                "path_params": ["request_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/{request_id}",
                "methods": ["GET"],
                "handler": "get_approval",
                "description": "获取审批记录",
                "body": False,
                "path_params": ["request_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/list",
                "methods": ["GET"],
                "handler": "list_approvals",
                "description": "列出审批记录 (支持过滤)",
                "body": False,
                "path_params": [],
                "query_params": ["status", "operation", "requester", "limit"],
            },
            {
                "path": "/cc2/approval/statistics",
                "methods": ["GET"],
                "handler": "get_approval_statistics",
                "description": "审批统计",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/trust-mode/activate",
                "methods": ["POST"],
                "handler": "activate_trust_mode",
                "description": "激活信任模式窗口",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/trust-mode/deactivate",
                "methods": ["POST"],
                "handler": "deactivate_trust_mode",
                "description": "停用信任模式",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/rule-preset/add",
                "methods": ["POST"],
                "handler": "add_rule_preset",
                "description": "添加规则预设",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/approval/rule-preset/{preset_id}",
                "methods": ["DELETE"],
                "handler": "remove_rule_preset",
                "description": "移除规则预设",
                "body": False,
                "path_params": ["preset_id"],
                "query_params": [],
            },
            # 3. 干预管理
            {
                "path": "/cc2/intervention/emergency-pause",
                "methods": ["POST"],
                "handler": "initiate_emergency_pause",
                "description": "发起紧急暂停",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/emergency-pause/{pause_id}/resolve",
                "methods": ["POST"],
                "handler": "resolve_emergency_pause",
                "description": "解决紧急暂停",
                "body": True,
                "path_params": ["pause_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/manual-override",
                "methods": ["POST"],
                "handler": "initiate_manual_override",
                "description": "发起人工接管",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/manual-override/{override_id}/release",
                "methods": ["POST"],
                "handler": "release_override",
                "description": "释放人工接管",
                "body": True,
                "path_params": ["override_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/correction",
                "methods": ["POST"],
                "handler": "submit_correction",
                "description": "提交纠正反馈",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/correction/{correction_id}/apply",
                "methods": ["POST"],
                "handler": "apply_correction",
                "description": "应用纠正",
                "body": True,
                "path_params": ["correction_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/creative-request",
                "methods": ["POST"],
                "handler": "request_creative_input",
                "description": "发起创意请求",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/creative-request/{request_id}/respond",
                "methods": ["POST"],
                "handler": "respond_creative_request",
                "description": "响应创意请求",
                "body": True,
                "path_params": ["request_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/active",
                "methods": ["GET"],
                "handler": "get_active_interventions",
                "description": "获取活跃干预",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/intervention/history",
                "methods": ["GET"],
                "handler": "get_intervention_history",
                "description": "获取干预历史",
                "body": False,
                "path_params": [],
                "query_params": ["limit", "event_type"],
            },
            {
                "path": "/cc2/intervention/statistics",
                "methods": ["GET"],
                "handler": "get_intervention_statistics",
                "description": "干预统计",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            # 4. 抗疲劳
            {
                "path": "/cc2/fatigue/statistics",
                "methods": ["GET"],
                "handler": "get_fatigue_statistics",
                "description": "抗疲劳统计",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/batch/add",
                "methods": ["POST"],
                "handler": "add_to_batch",
                "description": "添加批量审批项",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/batch/{user_id}/ready",
                "methods": ["GET"],
                "handler": "get_ready_batches",
                "description": "获取就绪批量组",
                "body": False,
                "path_params": ["user_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/batch/{batch_id}/resolve",
                "methods": ["POST"],
                "handler": "resolve_batch",
                "description": "解决批量组",
                "body": True,
                "path_params": ["batch_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/{user_id}/adjustment",
                "methods": ["GET"],
                "handler": "get_fatigue_adjustment",
                "description": "获取疲劳调整建议",
                "body": False,
                "path_params": ["user_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/{user_id}/trust",
                "methods": ["GET"],
                "handler": "get_trust_record",
                "description": "获取信任记录",
                "body": False,
                "path_params": ["user_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/{user_id}/trust/promote",
                "methods": ["POST"],
                "handler": "promote_trust",
                "description": "提升信任",
                "body": True,
                "path_params": ["user_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/{user_id}/trust/demote",
                "methods": ["POST"],
                "handler": "demote_trust",
                "description": "降低信任",
                "body": True,
                "path_params": ["user_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/fatigue/{user_id}",
                "methods": ["GET"],
                "handler": "get_fatigue_state",
                "description": "获取用户疲劳状态",
                "body": False,
                "path_params": ["user_id"],
                "query_params": [],
            },
            # 5. KPI 指标
            {
                "path": "/cc2/kpi/dashboard",
                "methods": ["GET"],
                "handler": "get_kpi_dashboard",
                "description": "KPI 仪表盘",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/kpi/alerts",
                "methods": ["GET"],
                "handler": "get_kpi_alerts",
                "description": "当前告警",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/kpi/{kpi_name}/summary",
                "methods": ["GET"],
                "handler": "get_kpi_summary",
                "description": "KPI 汇总",
                "body": False,
                "path_params": ["kpi_name"],
                "query_params": [],
            },
            {
                "path": "/cc2/kpi/{kpi_name}/history",
                "methods": ["GET"],
                "handler": "get_kpi_history",
                "description": "KPI 历史",
                "body": False,
                "path_params": ["kpi_name"],
                "query_params": ["limit"],
            },
            {
                "path": "/cc2/kpi/{kpi_name}/threshold",
                "methods": ["POST"],
                "handler": "adjust_kpi_threshold",
                "description": "调整 KPI 阈值",
                "body": True,
                "path_params": ["kpi_name"],
                "query_params": [],
            },
            # 6. 协作引擎
            {
                "path": "/cc2/agent/list",
                "methods": ["GET"],
                "handler": "list_agents",
                "description": "列出全部 Agent",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/agent/register",
                "methods": ["POST"],
                "handler": "register_agent",
                "description": "注册 Agent 配置",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/agent/{agent_id}/switch-mode",
                "methods": ["POST"],
                "handler": "switch_agent_mode",
                "description": "切换协作模式",
                "body": True,
                "path_params": ["agent_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/agent/{agent_id}/check-auto-step",
                "methods": ["POST"],
                "handler": "check_auto_step",
                "description": "检查自主步数",
                "body": True,
                "path_params": ["agent_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/agent/{agent_id}/interventions",
                "methods": ["GET"],
                "handler": "get_agent_interventions",
                "description": "获取 Agent 干预",
                "body": False,
                "path_params": ["agent_id"],
                "query_params": ["limit"],
            },
            {
                "path": "/cc2/agent/{agent_id}",
                "methods": ["GET"],
                "handler": "get_agent",
                "description": "获取 Agent 配置",
                "body": False,
                "path_params": ["agent_id"],
                "query_params": [],
            },
            {
                "path": "/cc2/agent/{agent_id}",
                "methods": ["PUT"],
                "handler": "update_agent",
                "description": "更新 Agent 配置",
                "body": True,
                "path_params": ["agent_id"],
                "query_params": [],
            },
            # 7. CC1 联动
            {
                "path": "/cc2/cc1/process-result",
                "methods": ["POST"],
                "handler": "process_cc1_result",
                "description": "处理 CC1 评审结果",
                "body": True,
                "path_params": [],
                "query_params": [],
            },
            # 8. 健康检查
            {
                "path": "/cc2/health",
                "methods": ["GET"],
                "handler": "health",
                "description": "健康检查",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
            {
                "path": "/cc2/health/ready",
                "methods": ["GET"],
                "handler": "health_ready",
                "description": "就绪检查 (含子系统状态)",
                "body": False,
                "path_params": [],
                "query_params": [],
            },
        ]

    def _create_async_handler(
        self,
        handler_name: str,
        *,
        has_body: bool,
        path_params: list[str],
        query_params: list[str],
    ):
        """为纯方法创建 Starlette 异步处理器.

        将 CC2APIRouter 的纯方法 (返回 dict) 适配为
        Starlette 异步处理器 (接收 Request, 返回 JSONResponse)。

        Args:
            handler_name: CC2APIRouter 上的方法名
            has_body: 是否需要解析 JSON 请求体
            path_params: 路径参数名列表
            query_params: 查询参数名列表

        Returns:
            async 函数 (Request → JSONResponse)
        """
        method = getattr(self, handler_name)

        async def handler(request: "Request") -> "JSONResponse":
            kwargs: dict[str, Any] = {}

            # 提取路径参数
            for param in path_params:
                if param in request.path_params:
                    kwargs[param] = request.path_params[param]

            # 提取请求体
            if has_body:
                try:
                    kwargs["body"] = await request.json()
                except Exception:
                    return JSONResponse(  # type: ignore[misc]
                        _err(400, "请求体解析失败")
                    )

            # 提取查询参数 (自动尝试 int 转换)
            for param in query_params:
                if param in request.query_params:
                    val = request.query_params[param]
                    try:
                        kwargs[param] = int(val)
                    except ValueError:
                        kwargs[param] = val

            try:
                result = method(**kwargs)
                return JSONResponse(result)  # type: ignore[misc]
            except Exception as exc:
                logger.exception(
                    "API 处理器异常: handler=%s error=%s",
                    handler_name, exc,
                )
                return JSONResponse(  # type: ignore[misc]
                    _err(500, f"内部错误: {exc}")
                )

        return handler

    def create_app(
        self,
        *,
        cors_origins: list[str] | None = None,
    ) -> "Starlette":
        """创建 Starlette 应用实例.

        将全部 CC2 端点注册为 Starlette 路由, 配置 CORS 中间件,
        返回可直接传给 ``uvicorn.run()`` 的应用实例。

        静态路由 (如 /list, /statistics) 在动态路由 (如 /{id}) 之前注册,
        确保 Starlette 路由匹配优先级正确。

        Args:
            cors_origins: CORS 允许的源列表 (默认 ["*"])

        Returns:
            配置好的 Starlette 应用

        Raises:
            ImportError: Starlette 未安装时

        Usage::

            router = CC2APIRouter(engine)
            app = router.create_app()
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        """
        if not _STARLETTE_AVAILABLE:
            raise ImportError(
                "Starlette 未安装, 无法创建应用。"
                "请安装: pip install starlette uvicorn"
            )

        origins = cors_origins if cors_origins is not None else ["*"]
        routes: list[Route] = []

        for route_def in self.get_routes():
            handler = self._create_async_handler(
                route_def["handler"],
                has_body=route_def["body"],
                path_params=route_def["path_params"],
                query_params=route_def["query_params"],
            )
            routes.append(
                Route(
                    route_def["path"],
                    handler,
                    methods=route_def["methods"],
                )
            )

        middleware: list[Middleware] = []
        if origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=origins,
                    allow_methods=(
                        ["*"]
                        if "*" in origins
                        else ["GET", "POST", "PUT", "DELETE"]
                    ),
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)
        return app
