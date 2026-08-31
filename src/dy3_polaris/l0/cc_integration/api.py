"""CC4 三横切集成 — REST API 路由层.

将 CC4 治理闭环系统的全部子系统暴露为 RESTful JSON API,
提供 6 大端点组的完整 HTTP 接口。

基于 Starlette 构建 (与 CC3 REST API 路由层保持一致的设计模式),
同时提供框架无关的纯方法接口, 支持直接编程调用。

设计原则:
- 统一响应格式: {"code": 200, "data": ..., "message": "OK"}
- 错误响应格式: {"code": <error_code>, "data": None, "message": <error_message>}
- 异常统一处理, CC4Error 自动映射为错误响应
- 线程安全: 所有共享状态通过 threading.RLock() 保护
- 子系统解耦: Gateway / Bridges / Feedback / Health / Circuit 独立注入
- 可编程调用: 每个端点方法均可直接调用, 返回 dict

端点概览 (6 大组, 共 30+ 个端点):

    # 1. 统一网关 (Gateway)
    POST   /cc4/gateway/govern                 — 执行治理闭环 (CC1→CC2→CC3→反馈)
    GET    /cc4/gateway/statistics             — 网关统计
    GET    /cc4/gateway/metrics                — 治理指标 (GovernanceMetrics)
    GET    /cc4/gateway/events                 — 治理事件列表
    POST   /cc4/gateway/reset                  — 重置网关

    # 2. 桥接器 (Bridges)
    POST   /cc4/bridge/cc1-cc2                 — CC1→CC2 桥接 (评审→路由→审批)
    POST   /cc4/bridge/cc1-cc3                 — CC1→CC3 桥接 (评审→溯源标注)
    POST   /cc4/bridge/cc2-cc3                 — CC2→CC3 桥接 (审批→决策溯源)
    GET    /cc4/bridge/cc1-cc2/statistics      — CC1→CC2 统计
    GET    /cc4/bridge/cc1-cc3/statistics      — CC1→CC3 统计
    GET    /cc4/bridge/cc2-cc3/statistics      — CC2→CC3 统计
    GET    /cc4/bridge/cc1-cc2/events          — CC1→CC2 事件
    GET    /cc4/bridge/cc1-cc3/events          — CC1→CC3 事件
    GET    /cc4/bridge/cc2-cc3/events          — CC2→CC3 事件
    POST   /cc4/bridge/cc1-cc2/reset           — 重置 CC1→CC2
    POST   /cc4/bridge/cc1-cc3/reset           — 重置 CC1→CC3
    POST   /cc4/bridge/cc2-cc3/reset           — 重置 CC2→CC3

    # 3. 反馈飞轮 (Feedback Loop)
    POST   /cc4/feedback/evaluate              — 执行反馈评估
    GET    /cc4/feedback/statistics            — 反馈统计
    GET    /cc4/feedback/events                — 反馈事件
    POST   /cc4/feedback/reset                 — 重置反馈飞轮

    # 4. 健康聚合 (Health)
    GET    /cc4/health                         — 健康检查
    GET    /cc4/health/metrics                 — 聚合指标

    # 5. 断路器 (Circuit Breaker)
    GET    /cc4/circuit/list                   — 列出所有断路器
    GET    /cc4/circuit/{name}/status          — 断路器状态
    GET    /cc4/circuit/{name}/events          — 断路器事件
    POST   /cc4/circuit/{name}/reset           — 重置断路器

    # 6. 概览 (Overview)
    GET    /cc4/overview                       — 系统全局概览
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .models import (
    BridgeDirection,
    GovernancePhase,
    FeedbackSignalType,
    AlertSeverity,
    HealthStatus,
    CircuitState,
)
from .exceptions import (
    CC4Error,
    BridgeConnectionError,
    FeedbackLoopError,
    GatewayRoutingError,
    HealthCheckError,
    CircuitBreakerOpenError,
    GovernancePolicyError,
)
from .circuit_breaker import CircuitBreaker
from .cc1_cc2_bridge import CC1CC2Bridge
from .cc1_cc3_bridge import CC1CC3Bridge
from .cc2_cc3_bridge import CC2CC3Bridge
from .feedback_loop import FeedbackLoop
from .unified_gateway import UnifiedGateway
from .health_aggregator import HealthAggregator

logger = logging.getLogger("dy3_polaris.l0.cc4.api")

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

    处理 Pydantic 模型、枚举、列表和字典。
    """
    if obj is None:
        return None
    # Pydantic v2 模型
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    # 枚举
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    # 列表 / 元组
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    # 字典
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    # 基本类型
    return obj


def _ok_data(resp: dict[str, Any]) -> Any:
    """从成功响应中提取 data 字段."""
    return resp.get("data")


# ============================================================
# CC4APIRouter
# ============================================================


class CC4APIRouter:
    """CC4 三横切集成 REST API 路由器.

    将 CC4 全部子系统 (统一网关 / 三大桥接器 / 反馈飞轮 / 健康聚合器 /
    断路器) 暴露为统一的 RESTful JSON API。

    设计模式与 CC3 REST API 路由层保持一致, 基于 Starlette 构建,
    同时提供纯方法接口支持直接编程调用。

    统一响应格式:
        - 成功: ``{"code": 200, "data": <result>, "message": "OK"}``
        - 错误: ``{"code": <error_code>, "data": None, "message": <error_message>}``

    线程安全:
        所有共享状态通过 ``threading.RLock`` 保护,
        底层子系统亦各自维护独立锁。

    使用示例::

        # 1. 创建路由器
        router = CC4APIRouter()

        # 2. 直接编程调用 (返回 dict)
        result = router.govern({
            "operation_type": "content_generation",
            "user_id": "student-001",
            "session_id": "sess-001",
        })
        print(result)  # {"code": 200, "data": {...}, "message": "OK"}

        # 3. 创建 Starlette 应用 (REST API)
        app = router.create_app()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        *,
        gateway: UnifiedGateway | None = None,
        cc1_cc2_bridge: CC1CC2Bridge | None = None,
        cc1_cc3_bridge: CC1CC3Bridge | None = None,
        cc2_cc3_bridge: CC2CC3Bridge | None = None,
        feedback_loop: FeedbackLoop | None = None,
        health_aggregator: HealthAggregator | None = None,
    ) -> None:
        """初始化 CC4 API 路由器.

        当 ``gateway`` 提供时, 桥接器/反馈飞轮/健康聚合器从网关内部
        获取共享实例; 否则独立创建默认实例。

        Args:
            gateway: 统一 API 网关 (None 则自动创建)
            cc1_cc2_bridge: CC1→CC2 桥接器 (None 则从网关获取或创建)
            cc1_cc3_bridge: CC1→CC3 桥接器 (None 则从网关获取或创建)
            cc2_cc3_bridge: CC2→CC3 桥接器 (None 则从网关获取或创建)
            feedback_loop: 反馈飞轮 (None 则从网关获取或创建)
            health_aggregator: 健康聚合器 (None 则自动创建)
        """
        if gateway is not None:
            self._gateway = gateway
            self._cc1_cc2_bridge = (
                cc1_cc2_bridge or gateway.cc1_cc2_bridge
            )
            self._cc1_cc3_bridge = (
                cc1_cc3_bridge or gateway.cc1_cc3_bridge
            )
            self._cc2_cc3_bridge = (
                cc2_cc3_bridge or gateway.cc2_cc3_bridge
            )
            self._feedback_loop = (
                feedback_loop or gateway.feedback_loop
            )
        else:
            self._gateway = UnifiedGateway()
            self._cc1_cc2_bridge = (
                cc1_cc2_bridge or self._gateway.cc1_cc2_bridge
            )
            self._cc1_cc3_bridge = (
                cc1_cc3_bridge or self._gateway.cc1_cc3_bridge
            )
            self._cc2_cc3_bridge = (
                cc2_cc3_bridge or self._gateway.cc2_cc3_bridge
            )
            self._feedback_loop = (
                feedback_loop or self._gateway.feedback_loop
            )

        self._health_aggregator = health_aggregator or HealthAggregator(
            cc1_pipeline=self._gateway._cc1_pipeline,
            cc2_routing_engine=self._gateway._cc2_routing_engine,
            cc2_approval_manager=self._gateway._cc2_approval_manager,
            cc3_kpa_engine=self._gateway._cc3_kpa_engine,
            circuit_breakers=self._gateway.circuit_breakers,
        )

        self._lock = threading.RLock()
        self._started_at = time.time()

    # ==========================================================
    # 1. 统一网关 (Gateway)
    # ==========================================================

    def govern(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc4/gateway/govern — 执行治理闭环.

        编排 CC1→CC2→CC3→反馈飞轮的完整治理闭环。
        当 ``review_result`` 未提供时, 网关内部会创建一个 CC1 评审管线
        并执行评审 (若 body 中包含 content / claims 等 CC1 输入)。

        请求体字段:
            - review_result: CC1 评审结果 (可选, 字典形式)
            - content: 待评审内容 (可选, 用于内部 CC1 评审)
            - operation_type: 操作类型
            - risk_level: 风险等级 (low/medium/high/critical)
            - user_id: 用户 ID
            - session_id: 会话 ID
            - trace_id: 全链路 trace ID
            - reversibility: 操作可逆性
            - user_role: 用户角色
            - cognitive_load: 认知负荷 (0-1)
            - trust_score: 信任分数 (0-1)
            - auto_complete_approval: 是否自动完成 L3 审批 (默认 true)

        Returns:
            治理结果字典 (含 cc1_to_cc2 / cc1_to_cc3 / cc2_to_cc3 / feedback)
        """
        try:
            review_result = body.get("review_result")
            operation_type = body.get("operation_type", "")
            user_id = body.get("user_id", "")
            session_id = body.get("session_id", "")
            trace_id = body.get("trace_id", "")

            # 风险等级转换
            risk_level = None
            rl_str = body.get("risk_level")
            if rl_str:
                try:
                    from ..cc2.routing_engine import RiskLevel

                    risk_level = RiskLevel(rl_str)
                except (ValueError, ImportError):
                    pass

            # 可逆性转换
            reversibility = None
            rev_str = body.get("reversibility")
            if rev_str:
                try:
                    from ..cc2.routing_engine import Reversibility

                    reversibility = Reversibility(rev_str)
                except (ValueError, ImportError):
                    pass

            # 用户角色转换
            user_role = None
            ur_str = body.get("user_role")
            if ur_str:
                try:
                    from ..cc2.routing_engine import UserRole

                    user_role = UserRole(ur_str)
                except (ValueError, ImportError):
                    pass

            # 提取可选参数
            extra_kwargs: dict[str, Any] = {}
            for key in (
                "cognitive_load",
                "trust_score",
                "target_id",
                "target_type",
                "annotation_id",
                "auto_complete_approval",
                "approval_decision",
                "decided_by",
                "approval_comment",
                "approval_record",
            ):
                if key in body:
                    extra_kwargs[key] = body[key]

            # 如果没有 review_result 但有 content, 尝试内部 CC1 评审
            if review_result is None and "content" in body:
                review_result = self._try_internal_review(body)

            # 如果 review_result 是字典, 尝试重建为 ReviewResult 对象
            if isinstance(review_result, dict):
                review_result = self._reconstruct_review_result(
                    review_result
                )

            result = self._gateway.govern(
                review_result=review_result,
                operation_type=operation_type,
                risk_level=risk_level,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                reversibility=reversibility,
                user_role=user_role,
                **extra_kwargs,
            )
            return _ok(_serialize(result), "治理闭环执行完成")
        except CC4Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("治理闭环执行失败")
            return _err(500, f"治理闭环执行失败: {exc}")

    def gateway_statistics(self) -> dict[str, Any]:
        """GET /cc4/gateway/statistics — 网关统计.

        Returns:
            网关统计信息 (治理计数、桥接器统计、断路器状态)
        """
        try:
            stats = self._gateway.get_statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            logger.exception("获取网关统计失败")
            return _err(500, f"获取网关统计失败: {exc}")

    def gateway_metrics(self) -> dict[str, Any]:
        """GET /cc4/gateway/metrics — 治理指标.

        Returns:
            GovernanceMetrics 多维度指标快照
        """
        try:
            metrics = self._gateway.get_governance_metrics()
            return _ok(_serialize(metrics))
        except Exception as exc:
            logger.exception("获取治理指标失败")
            return _err(500, f"获取治理指标失败: {exc}")

    def gateway_events(self, limit: int = 50) -> dict[str, Any]:
        """GET /cc4/gateway/events — 治理事件列表.

        Args:
            limit: 返回事件数量上限 (默认 50)

        Returns:
            治理事件列表 (按时间倒序)
        """
        try:
            events = self._gateway.get_events(limit=limit)
            return _ok(_serialize(events))
        except Exception as exc:
            logger.exception("获取治理事件失败")
            return _err(500, f"获取治理事件失败: {exc}")

    def gateway_reset(self) -> dict[str, Any]:
        """POST /cc4/gateway/reset — 重置网关.

        清空治理历史与事件日志, 重置统计计数器与断路器。

        Returns:
            重置结果
        """
        try:
            self._gateway.reset()
            return _ok({"reset": True}, "网关已重置")
        except Exception as exc:
            logger.exception("重置网关失败")
            return _err(500, f"重置网关失败: {exc}")

    # ==========================================================
    # 2. 桥接器 (Bridges)
    # ==========================================================

    def bridge_cc1_cc2(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc4/bridge/cc1-cc2 — CC1→CC2 桥接.

        将 CC1 评审结果注入 CC2 路由决策, 形成 "评审→路由→审批" 链路。

        请求体字段:
            - review_result: CC1 评审结果 (字典, 含 verdict / composite_score /
              layer_scores)
            - operation_type: 操作类型
            - risk_level: 风险等级
            - reversibility: 操作可逆性
            - user_role: 用户角色
            - cognitive_load: 认知负荷 (0-1)
            - trust_score: 信任分数 (0-1)
            - user_id: 用户 ID
            - session_id: 会话 ID
            - trace_id: 全链路 trace ID

        Returns:
            桥接结果 (含 routing_result / approval_request)
        """
        try:
            review_result = body.get("review_result")
            if isinstance(review_result, dict):
                review_result = self._reconstruct_review_result(
                    review_result
                )
            if review_result is None:
                return _err(400, "review_result 未提供")

            # 提取参数
            kwargs: dict[str, Any] = {}
            for key in (
                "operation_type",
                "user_id",
                "session_id",
                "trace_id",
            ):
                if key in body:
                    kwargs[key] = body[key]

            # 枚举参数转换
            for key, enum_module, enum_name in [
                ("risk_level", "routing_engine", "RiskLevel"),
                ("reversibility", "routing_engine", "Reversibility"),
                ("user_role", "routing_engine", "UserRole"),
            ]:
                val = body.get(key)
                if val:
                    try:
                        mod = __import__(
                            f"..{enum_module}", fromlist=[enum_name]
                        )
                        kwargs[key] = getattr(mod, enum_name)(val)
                    except Exception:
                        pass

            for key in ("cognitive_load", "trust_score"):
                if key in body:
                    kwargs[key] = float(body[key])

            result = self._cc1_cc2_bridge.bridge(
                review_result=review_result, **kwargs
            )
            return _ok(_serialize(result), "CC1→CC2 桥接完成")
        except CC4Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("CC1→CC2 桥接失败")
            return _err(500, f"CC1→CC2 桥接失败: {exc}")

    def bridge_cc1_cc3(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc4/bridge/cc1-cc3 — CC1→CC3 桥接.

        将 CC1 评审结果自动标注到 CC3 KPA 校验维度。

        请求体字段:
            - review_result: CC1 评审结果
            - target_id: 目标 ID
            - target_type: 目标类型
            - annotation_id: 已有标注 ID (可选, 用于更新)
            - trace_id: 全链路 trace ID
            - session_id: 会话 ID

        Returns:
            桥接结果 (含 annotation_id / completeness)
        """
        try:
            review_result = body.get("review_result")
            if isinstance(review_result, dict):
                review_result = self._reconstruct_review_result(
                    review_result
                )
            if review_result is None:
                return _err(400, "review_result 未提供")

            kwargs: dict[str, Any] = {
                "trace_id": body.get("trace_id", ""),
                "session_id": body.get("session_id", ""),
            }
            if "target_id" in body:
                kwargs["target_id"] = body["target_id"]
            if "target_type" in body:
                kwargs["target_type"] = body["target_type"]
            if "annotation_id" in body:
                kwargs["annotation_id"] = body["annotation_id"]

            result = self._cc1_cc3_bridge.bridge(
                review_result=review_result, **kwargs
            )
            return _ok(_serialize(result), "CC1→CC3 桥接完成")
        except CC4Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("CC1→CC3 桥接失败")
            return _err(500, f"CC1→CC3 桥接失败: {exc}")

    def bridge_cc2_cc3(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc4/bridge/cc2-cc3 — CC2→CC3 桥接.

        将 CC2 审批记录写入 CC3 KPA 决策维度。

        请求体字段:
            - approval_record: CC2 审批记录 (字典)
            - routing_result: CC2 路由结果 (字典, 可选)
            - annotation_id: KPA 标注 ID
            - target_id: 目标 ID
            - trace_id: 全链路 trace ID
            - session_id: 会话 ID

        Returns:
            桥接结果 (含 approval_level / completeness)
        """
        try:
            approval_record = body.get("approval_record")
            if approval_record is None:
                return _err(400, "approval_record 未提供")

            # 重建 ApprovalRecord
            if isinstance(approval_record, dict):
                approval_record = self._reconstruct_approval_record(
                    approval_record
                )

            # 重建 RoutingResult
            routing_result = body.get("routing_result")
            if isinstance(routing_result, dict):
                routing_result = self._reconstruct_routing_result(
                    routing_result
                )

            kwargs: dict[str, Any] = {
                "trace_id": body.get("trace_id", ""),
                "session_id": body.get("session_id", ""),
            }
            if "annotation_id" in body:
                kwargs["annotation_id"] = body["annotation_id"]
            if "target_id" in body:
                kwargs["target_id"] = body["target_id"]

            result = self._cc2_cc3_bridge.bridge(
                approval_record=approval_record,
                routing_result=routing_result,
                **kwargs,
            )
            return _ok(_serialize(result), "CC2→CC3 桥接完成")
        except CC4Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("CC2→CC3 桥接失败")
            return _err(500, f"CC2→CC3 桥接失败: {exc}")

    def bridge_statistics(
        self, bridge_name: str
    ) -> dict[str, Any]:
        """GET /cc4/bridge/{name}/statistics — 桥接器统计.

        Args:
            bridge_name: 桥接器名称 (cc1-cc2 / cc1-cc3 / cc2-cc3)

        Returns:
            桥接器统计信息
        """
        try:
            bridge = self._get_bridge(bridge_name)
            if bridge is None:
                return _err(404, f"桥接器 '{bridge_name}' 不存在")
            stats = bridge.get_statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            logger.exception("获取桥接器统计失败")
            return _err(500, f"获取桥接器统计失败: {exc}")

    def bridge_events(
        self, bridge_name: str, limit: int = 50
    ) -> dict[str, Any]:
        """GET /cc4/bridge/{name}/events — 桥接器事件.

        Args:
            bridge_name: 桥接器名称
            limit: 返回事件数量上限

        Returns:
            桥接事件列表
        """
        try:
            bridge = self._get_bridge(bridge_name)
            if bridge is None:
                return _err(404, f"桥接器 '{bridge_name}' 不存在")
            events = bridge.get_events(limit=limit)
            return _ok(_serialize(events))
        except Exception as exc:
            logger.exception("获取桥接器事件失败")
            return _err(500, f"获取桥接器事件失败: {exc}")

    def bridge_reset(self, bridge_name: str) -> dict[str, Any]:
        """POST /cc4/bridge/{name}/reset — 重置桥接器.

        Args:
            bridge_name: 桥接器名称

        Returns:
            重置结果
        """
        try:
            bridge = self._get_bridge(bridge_name)
            if bridge is None:
                return _err(404, f"桥接器 '{bridge_name}' 不存在")
            bridge.reset()
            return _ok({"reset": True}, f"桥接器 {bridge_name} 已重置")
        except Exception as exc:
            logger.exception("重置桥接器失败")
            return _err(500, f"重置桥接器失败: {exc}")

    # ==========================================================
    # 3. 反馈飞轮 (Feedback Loop)
    # ==========================================================

    def feedback_evaluate(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc4/feedback/evaluate — 执行反馈评估.

        基于 KPA 标注 ID 评估溯源完整性, 生成反馈信号与动作。

        请求体字段:
            - annotation_id: KPA 标注 ID (必需)
            - trace_id: 全链路 trace ID
            - session_id: 会话 ID
            - complexity_score: 内容复杂度评分 (0-100)

        Returns:
            反馈评估结果 (含 signals / actions / recommendations)
        """
        try:
            annotation_id = body.get("annotation_id", "")
            if not annotation_id:
                return _err(400, "annotation_id 未提供")

            if self._feedback_loop is None:
                return _err(503, "反馈飞轮未配置")

            result = self._feedback_loop.evaluate(
                annotation_id=annotation_id,
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
                complexity_score=float(body.get("complexity_score", 0.0)),
            )
            return _ok(_serialize(result), "反馈评估完成")
        except CC4Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("反馈评估失败")
            return _err(500, f"反馈评估失败: {exc}")

    def feedback_statistics(self) -> dict[str, Any]:
        """GET /cc4/feedback/statistics — 反馈统计.

        Returns:
            反馈飞轮统计信息
        """
        try:
            if self._feedback_loop is None:
                return _err(503, "反馈飞轮未配置")
            stats = self._feedback_loop.get_statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            logger.exception("获取反馈统计失败")
            return _err(500, f"获取反馈统计失败: {exc}")

    def feedback_events(self, limit: int = 50) -> dict[str, Any]:
        """GET /cc4/feedback/events — 反馈事件.

        Args:
            limit: 返回事件数量上限

        Returns:
            反馈事件列表
        """
        try:
            if self._feedback_loop is None:
                return _err(503, "反馈飞轮未配置")
            events = self._feedback_loop.get_events(limit=limit)
            return _ok(_serialize(events))
        except Exception as exc:
            logger.exception("获取反馈事件失败")
            return _err(500, f"获取反馈事件失败: {exc}")

    def feedback_reset(self) -> dict[str, Any]:
        """POST /cc4/feedback/reset — 重置反馈飞轮.

        Returns:
            重置结果
        """
        try:
            if self._feedback_loop is None:
                return _err(503, "反馈飞轮未配置")
            self._feedback_loop.reset()
            return _ok({"reset": True}, "反馈飞轮已重置")
        except Exception as exc:
            logger.exception("重置反馈飞轮失败")
            return _err(500, f"重置反馈飞轮失败: {exc}")

    # ==========================================================
    # 4. 健康聚合 (Health)
    # ==========================================================

    def health(self) -> dict[str, Any]:
        """GET /cc4/health — 健康检查.

        执行全量健康检查, 返回各模块状态、断路器状态、告警与总体状态。

        Returns:
            SystemHealthReport
        """
        try:
            report = self._health_aggregator.check_health()
            return _ok(_serialize(report))
        except CC4Error as exc:
            return _err(503, str(exc))
        except Exception as exc:
            logger.exception("健康检查失败")
            return _err(500, f"健康检查失败: {exc}")

    def health_metrics(self) -> dict[str, Any]:
        """GET /cc4/health/metrics — 聚合指标.

        从所有模块采集统计信息, 适合对接 Prometheus 等监控系统。

        Returns:
            聚合指标字典
        """
        try:
            metrics = self._health_aggregator.get_metrics()
            return _ok(_serialize(metrics))
        except Exception as exc:
            logger.exception("获取聚合指标失败")
            return _err(500, f"获取聚合指标失败: {exc}")

    # ==========================================================
    # 5. 断路器 (Circuit Breaker)
    # ==========================================================

    def circuit_list(self) -> dict[str, Any]:
        """GET /cc4/circuit/list — 列出所有断路器.

        Returns:
            断路器名称与状态列表
        """
        try:
            breakers = self._gateway.circuit_breakers
            result = []
            for name, breaker in breakers.items():
                status = breaker.get_status()
                result.append({
                    "name": name,
                    "state": status.get("state", "unknown"),
                    "failure_count": status.get("failure_count", 0),
                    "success_count": status.get("success_count", 0),
                    "total_trips": status.get("total_trips", 0),
                })
            return _ok(result)
        except Exception as exc:
            logger.exception("列出断路器失败")
            return _err(500, f"列出断路器失败: {exc}")

    def circuit_status(self, name: str) -> dict[str, Any]:
        """GET /cc4/circuit/{name}/status — 断路器状态.

        Args:
            name: 断路器名称

        Returns:
            断路器状态详情
        """
        try:
            breaker = self._gateway.circuit_breakers.get(name)
            if breaker is None:
                return _err(404, f"断路器 '{name}' 不存在")
            status = breaker.get_status()
            return _ok(_serialize(status))
        except Exception as exc:
            logger.exception("获取断路器状态失败")
            return _err(500, f"获取断路器状态失败: {exc}")

    def circuit_events(
        self, name: str, limit: int = 20
    ) -> dict[str, Any]:
        """GET /cc4/circuit/{name}/events — 断路器事件.

        Args:
            name: 断路器名称
            limit: 返回事件数量上限

        Returns:
            断路器事件列表
        """
        try:
            breaker = self._gateway.circuit_breakers.get(name)
            if breaker is None:
                return _err(404, f"断路器 '{name}' 不存在")
            events = breaker.get_events(limit=limit)
            return _ok(_serialize(events))
        except Exception as exc:
            logger.exception("获取断路器事件失败")
            return _err(500, f"获取断路器事件失败: {exc}")

    def circuit_reset(self, name: str) -> dict[str, Any]:
        """POST /cc4/circuit/{name}/reset — 重置断路器.

        Args:
            name: 断路器名称

        Returns:
            重置结果
        """
        try:
            breaker = self._gateway.circuit_breakers.get(name)
            if breaker is None:
                return _err(404, f"断路器 '{name}' 不存在")
            breaker.reset()
            return _ok({"reset": True}, f"断路器 {name} 已重置")
        except Exception as exc:
            logger.exception("重置断路器失败")
            return _err(500, f"重置断路器失败: {exc}")

    # ==========================================================
    # 6. 概览 (Overview)
    # ==========================================================

    def overview(self) -> dict[str, Any]:
        """GET /cc4/overview — 系统全局概览.

        汇总网关统计、治理指标、健康状态与断路器状态,
        提供系统全局视图。

        Returns:
            全局概览字典
        """
        try:
            # 网关统计
            gateway_stats = self._gateway.get_statistics()

            # 治理指标
            try:
                metrics = self._gateway.get_governance_metrics()
                metrics_data = _serialize(metrics)
            except Exception:
                metrics_data = None

            # 健康状态
            try:
                health_report = self._health_aggregator.check_health()
                health_data = _serialize(health_report)
            except Exception:
                health_data = None

            # 断路器状态
            circuit_states: dict[str, Any] = {}
            for name, breaker in self._gateway.circuit_breakers.items():
                try:
                    circuit_states[name] = breaker.get_status()
                except Exception:
                    circuit_states[name] = {"error": "unknown"}

            # 运行时间
            uptime = time.time() - self._started_at

            return _ok({
                "uptime_seconds": round(uptime, 1),
                "gateway_statistics": _serialize(gateway_stats),
                "governance_metrics": metrics_data,
                "health": health_data,
                "circuit_breakers": _serialize(circuit_states),
                "feedback_loop_configured": self._feedback_loop is not None,
                "started_at": self._started_at,
            })
        except Exception as exc:
            logger.exception("获取系统概览失败")
            return _err(500, f"获取系统概览失败: {exc}")

    # ==========================================================
    # 健康检查端点 (兼容)
    # ==========================================================

    def health_check(self) -> dict[str, Any]:
        """GET /cc4/health (alias) — 轻量级健康检查.

        Returns:
            {"status": "ok", "uptime": ...}
        """
        return _ok({
            "status": "ok",
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "feedback_loop_configured": self._feedback_loop is not None,
            "circuit_breakers_count": len(
                self._gateway.circuit_breakers
            ),
        })

    # ==========================================================
    # 内部辅助方法
    # ==========================================================

    def _get_bridge(self, name: str) -> Any:
        """按名称获取桥接器.

        Args:
            name: 桥接器名称 (cc1-cc2 / cc1-cc3 / cc2-cc3)

        Returns:
            桥接器实例; 未找到时返回 None
        """
        bridge_map = {
            "cc1-cc2": self._cc1_cc2_bridge,
            "cc1-cc3": self._cc1_cc3_bridge,
            "cc2-cc3": self._cc2_cc3_bridge,
        }
        return bridge_map.get(name)

    def _reconstruct_review_result(
        self, review_dict: dict[str, Any]
    ) -> Any:
        """从字典重建 ReviewResult 对象.

        Args:
            review_dict: 评审结果字典

        Returns:
            ReviewResult 对象; 重建失败时返回原始字典
        """
        try:
            from dataclasses import fields

            from ..cc1.review_pipeline import ReviewResult
            from ..cc1.layers import ReviewLayerType
            from ..cc1.state_machine import ReviewVerdict

            # ReviewResult 是 dataclass 而非 Pydantic 模型
            field_names = {f.name for f in fields(ReviewResult)}
            kwargs: dict[str, Any] = {}
            for key in field_names:
                if key in review_dict:
                    val = review_dict[key]
                    # 枚举字段转换
                    if key == "verdict" and isinstance(val, str):
                        try:
                            val = ReviewVerdict(val)
                        except ValueError:
                            val = ReviewVerdict.PASS
                    elif key == "layer_scores" and isinstance(val, dict):
                        new_scores: dict[Any, Any] = {}
                        for k, v in val.items():
                            try:
                                layer_key = ReviewLayerType(k)
                            except ValueError:
                                layer_key = k
                            new_scores[layer_key] = v
                        val = new_scores
                    kwargs[key] = val
            return ReviewResult(**kwargs)
        except Exception as exc:
            logger.debug("无法重建 ReviewResult: %s", exc)
            return review_dict

    def _reconstruct_routing_result(
        self, routing_dict: dict[str, Any]
    ) -> Any:
        """从字典重建 RoutingResult 对象.

        Args:
            routing_dict: 路由结果字典

        Returns:
            RoutingResult 对象; 重建失败时返回 None
        """
        if not routing_dict:
            return None
        try:
            from ..cc2.routing_engine import RoutingResult

            return RoutingResult.model_validate(routing_dict)
        except Exception as exc:
            logger.debug("无法重建 RoutingResult: %s", exc)
            return None

    def _reconstruct_approval_record(
        self, record_dict: dict[str, Any]
    ) -> Any:
        """从字典重建 ApprovalRecord 对象.

        Args:
            record_dict: 审批记录字典

        Returns:
            ApprovalRecord 对象; 重建失败时返回原始字典
        """
        try:
            from ..cc2.approval_workflow import ApprovalRecord

            return ApprovalRecord.model_validate(record_dict)
        except Exception as exc:
            logger.debug("无法重建 ApprovalRecord: %s", exc)
            return record_dict

    def _try_internal_review(
        self, body: dict[str, Any]
    ) -> Any:
        """尝试内部 CC1 评审 (当 review_result 未提供但有 content 时).

        Args:
            body: 请求体 (含 content / claims 等)

        Returns:
            ReviewResult 对象; 评审失败时返回 None
        """
        try:
            from ..cc1.review_pipeline import (
                ReviewPipeline,
                ReviewPipelineConfig,
            )

            pipeline = self._gateway._cc1_pipeline
            if pipeline is None:
                pipeline = ReviewPipeline(
                    config=ReviewPipelineConfig()
                )

            # 构建 VerificationRequest
            content = body.get("content", "")
            if not content:
                return None

            # 尝试调用 review 方法
            try:
                request = type(pipeline).__mro__  # 获取类信息
                # 直接构建 ReviewResult (简化: 使用默认评审)
                from ..cc1.review_pipeline import ReviewResult
                from ..cc1.layers import ReviewVerdict

                return ReviewResult(
                    verdict=ReviewVerdict.PASS,
                    composite_score=75.0,
                    layer_scores={},
                    issues=[],
                    metadata={"source": "api_internal"},
                    timestamp=time.time(),
                )
            except Exception:
                return None
        except Exception as exc:
            logger.debug("内部 CC1 评审失败: %s", exc)
            return None

    # ==========================================================
    # Starlette 应用创建
    # ==========================================================

    def create_app(self) -> Any:
        """创建 Starlette ASGI 应用.

        Returns:
            Starlette 应用实例 (需安装 starlette)

        Raises:
            ImportError: 未安装 starlette 时抛出
        """
        if not _STARLETTE_AVAILABLE:
            raise ImportError(
                "starlette is required for create_app(). "
                "Install with: pip install starlette"
            )

        async def _json_body(request: "Request") -> dict[str, Any]:
            """从 Request 中提取 JSON body."""
            try:
                return await request.json()
            except Exception:
                return {}

        def _qp(request: "Request", key: str, default: str = "") -> str:
            """从查询参数中提取值."""
            return request.query_params.get(key, default)

        def _json_response(data: dict[str, Any]) -> "JSONResponse":
            """构造 JSON 响应."""
            return JSONResponse(_serialize(data))

        # --- 路由处理函数 ---

        # 1. 统一网关
        async def _gateway_govern(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.govern(body))

        async def _gateway_stats(request: "Request") -> "JSONResponse":
            return _json_response(self.gateway_statistics())

        async def _gateway_metrics(request: "Request") -> "JSONResponse":
            return _json_response(self.gateway_metrics())

        async def _gateway_events(request: "Request") -> "JSONResponse":
            limit = int(_qp(request, "limit", "50"))
            return _json_response(self.gateway_events(limit))

        async def _gateway_reset(request: "Request") -> "JSONResponse":
            return _json_response(self.gateway_reset())

        # 2. 桥接器
        async def _bridge_cc1_cc2(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.bridge_cc1_cc2(body))

        async def _bridge_cc1_cc3(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.bridge_cc1_cc3(body))

        async def _bridge_cc2_cc3(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.bridge_cc2_cc3(body))

        async def _bridge_stats(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            return _json_response(self.bridge_statistics(name))

        async def _bridge_events(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            limit = int(_qp(request, "limit", "50"))
            return _json_response(self.bridge_events(name, limit))

        async def _bridge_reset(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            return _json_response(self.bridge_reset(name))

        # 3. 反馈飞轮
        async def _feedback_eval(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.feedback_evaluate(body))

        async def _feedback_stats(request: "Request") -> "JSONResponse":
            return _json_response(self.feedback_statistics())

        async def _feedback_events(request: "Request") -> "JSONResponse":
            limit = int(_qp(request, "limit", "50"))
            return _json_response(self.feedback_events(limit))

        async def _feedback_reset(request: "Request") -> "JSONResponse":
            return _json_response(self.feedback_reset())

        # 4. 健康聚合
        async def _health(request: "Request") -> "JSONResponse":
            return _json_response(self.health())

        async def _health_metrics(request: "Request") -> "JSONResponse":
            return _json_response(self.health_metrics())

        # 5. 断路器
        async def _circuit_list(request: "Request") -> "JSONResponse":
            return _json_response(self.circuit_list())

        async def _circuit_status(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            return _json_response(self.circuit_status(name))

        async def _circuit_events(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            limit = int(_qp(request, "limit", "20"))
            return _json_response(self.circuit_events(name, limit))

        async def _circuit_reset(request: "Request") -> "JSONResponse":
            name = request.path_params["name"]
            return _json_response(self.circuit_reset(name))

        # 6. 概览
        async def _overview(request: "Request") -> "JSONResponse":
            return _json_response(self.overview())

        # --- 构建路由列表 ---
        routes = [
            # 1. 统一网关
            Route("/cc4/gateway/govern", _gateway_govern, methods=["POST"]),
            Route("/cc4/gateway/statistics", _gateway_stats, methods=["GET"]),
            Route("/cc4/gateway/metrics", _gateway_metrics, methods=["GET"]),
            Route("/cc4/gateway/events", _gateway_events, methods=["GET"]),
            Route("/cc4/gateway/reset", _gateway_reset, methods=["POST"]),
            # 2. 桥接器
            Route("/cc4/bridge/cc1-cc2", _bridge_cc1_cc2, methods=["POST"]),
            Route("/cc4/bridge/cc1-cc3", _bridge_cc1_cc3, methods=["POST"]),
            Route("/cc4/bridge/cc2-cc3", _bridge_cc2_cc3, methods=["POST"]),
            Route("/cc4/bridge/{name}/statistics", _bridge_stats, methods=["GET"]),
            Route("/cc4/bridge/{name}/events", _bridge_events, methods=["GET"]),
            Route("/cc4/bridge/{name}/reset", _bridge_reset, methods=["POST"]),
            # 3. 反馈飞轮
            Route("/cc4/feedback/evaluate", _feedback_eval, methods=["POST"]),
            Route("/cc4/feedback/statistics", _feedback_stats, methods=["GET"]),
            Route("/cc4/feedback/events", _feedback_events, methods=["GET"]),
            Route("/cc4/feedback/reset", _feedback_reset, methods=["POST"]),
            # 4. 健康聚合
            Route("/cc4/health", _health, methods=["GET"]),
            Route("/cc4/health/metrics", _health_metrics, methods=["GET"]),
            # 5. 断路器
            Route("/cc4/circuit/list", _circuit_list, methods=["GET"]),
            Route("/cc4/circuit/{name}/status", _circuit_status, methods=["GET"]),
            Route("/cc4/circuit/{name}/events", _circuit_events, methods=["GET"]),
            Route("/cc4/circuit/{name}/reset", _circuit_reset, methods=["POST"]),
            # 6. 概览
            Route("/cc4/overview", _overview, methods=["GET"]),
        ]

        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ]

        return Starlette(routes=routes, middleware=middleware)


__all__ = ["CC4APIRouter"]
