"""CC3 溯源捕获层 — REST API 路由层.

将 CC3 溯源捕获系统 (Provenance Capture) 的全部子系统暴露为
RESTful JSON API, 提供 9 大端点组的完整 HTTP 接口。

基于 Starlette 构建 (与 L6 REST API 路由层保持一致的设计模式),
同时提供框架无关的纯方法接口, 支持直接编程调用。

设计原则:
- 统一响应格式: {"code": 200, "data": ..., "message": "OK"}
- 错误响应格式: {"code": <error_code>, "data": None, "message": <error_message>}
- 异常统一处理, CC3Error 自动映射为错误响应
- 线程安全: 所有共享状态通过 threading.RLock() 保护
- 子系统解耦: KPA / DL / Chain / Ledger / Query / Integration / Metrics / Visualizer 独立注入
- 可编程调用: 每个端点方法均可直接调用, 返回 dict

端点概览 (9 大组, 共 60+ 个端点):

    # 1. KPA 标注 (Annotation)
    POST   /cc3/annotation/create              — 创建 KPA 标注
    GET    /cc3/annotation/{annotation_id}      — 获取标注详情
    GET    /cc3/annotation/list                 — 列出标注
    POST   /cc3/annotation/{annotation_id}/dimension — 更新维度
    POST   /cc3/annotation/{annotation_id}/validation — 更新校验维度
    POST   /cc3/annotation/{annotation_id}/decision — 更新决策维度
    POST   /cc3/annotation/{annotation_id}/propagation — 记录传播
    POST   /cc3/annotation/{annotation_id}/rules — 应用 Dy3+ 规则
    GET    /cc3/annotation/{annotation_id}/completeness — 评估完整度
    GET    /cc3/annotation/{annotation_id}/verify-hash — 哈希校验
    GET    /cc3/annotation/{annotation_id}/verify-signature — 签名校验
    GET    /cc3/annotation/{annotation_id}/prov — W3C PROV 映射
    GET    /cc3/annotation/statistics           — 标注统计

    # 2. 辩论日志 (Debate Log)
    POST   /cc3/debate/create                   — 创建辩论日志
    POST   /cc3/debate/{log_id}/round           — 追加辩论轮次
    POST   /cc3/debate/{log_id}/convergence     — 检查收敛
    POST   /cc3/debate/{log_id}/adjudication    — 记录裁决
    POST   /cc3/debate/{log_id}/outcome         — 记录结果
    POST   /cc3/debate/{log_id}/resource        — 记录资源消耗
    POST   /cc3/debate/{log_id}/finalize        — 完成日志
    GET    /cc3/debate/{log_id}/verify          — 完整性校验
    GET    /cc3/debate/{log_id}/export          — 导出日志
    GET    /cc3/debate/{log_id}                 — 获取日志
    GET    /cc3/debate/list                     — 列出日志
    GET    /cc3/debate/statistics               — 辩论统计

    # 3. 溯源链 (Chain)
    POST   /cc3/chain/create                    — 创建溯源链
    POST   /cc3/chain/{chain_id}/append         — 追加节点
    GET    /cc3/chain/{chain_id}/verify         — 验证链完整性
    GET    /cc3/chain/{chain_id}/verify-node    — 验证单个节点
    POST   /cc3/chain/{chain_id}/merkle         — 构建 Merkle 树
    GET    /cc3/chain/{chain_id}/merkle-proof   — 获取 Merkle 证明
    POST   /cc3/chain/{chain_id}/compress       — 压缩链
    GET    /cc3/chain/{chain_id}/snapshot       — 创建快照
    POST   /cc3/chain/audit-verify              — 审计验证
    GET    /cc3/chain/list                      — 列出溯源链
    POST   /cc3/chain/trace-cross-layer         — 跨层追踪

    # 4. L0 Ledger (Ledger)
    POST   /cc3/ledger/write-kpa                — 写入 KPA 事件
    POST   /cc3/ledger/write-dl                 — 写入 DL 事件
    POST   /cc3/ledger/write-cross-layer        — 写入跨层事件
    POST   /cc3/ledger/write-human-override     — 写入人工干预
    GET    /cc3/ledger/event/{event_id}         — 获取事件
    GET    /cc3/ledger/query                    — 查询事件
    GET    /cc3/ledger/query-by-time            — 按时间范围查询
    GET    /cc3/ledger/verify                   — 验证 Ledger
    GET    /cc3/ledger/statistics               — Ledger 统计

    # 5. 查询引擎 (Query)
    GET    /cc3/query/trace/{trace_id}          — 按 trace_id 回溯
    GET    /cc3/query/knowledge/{target_id}     — 知识溯源档案
    GET    /cc3/query/agent/{agent_id}          — Agent 操作历史
    GET    /cc3/query/timeline                  — 时间线查询
    GET    /cc3/query/graph                     — 溯源图
    GET    /cc3/query/overview                  — 全局概览

    # 6. CC1/CC2 集成 (Integration)
    POST   /cc3/integration/cc1-review          — CC1 评审回调
    POST   /cc3/integration/cc2-approval        — CC2 审批回调
    POST   /cc3/integration/check-provenance    — 溯源检查 (CC1)
    POST   /cc3/integration/check-escalation    — 升级检查 (CC2)
    POST   /cc3/integration/check-debate        — 辩论触发检查

    # 7. KPI 指标 (Metrics)
    GET    /cc3/metrics/dashboard               — KPI 仪表盘
    GET    /cc3/metrics/collect-all             — 全量采集
    GET    /cc3/metrics/coverage                — 覆盖率采集
    GET    /cc3/metrics/integrity               — 完整性采集
    GET    /cc3/metrics/performance             — 性能采集
    GET    /cc3/metrics/compliance              — 合规性采集
    GET    /cc3/metrics/export-dashboard        — 导出仪表盘

    # 8. 可视化 (Visualization)
    GET    /cc3/viz/cytoscape                   — Cytoscape.js 图数据
    GET    /cc3/viz/d3-hierarchy                — D3.js 层级树
    GET    /cc3/viz/mermaid                     — Mermaid 流程图
    GET    /cc3/viz/echarts-timeline            — ECharts 辩论时间线
    GET    /cc3/viz/echarts-radar               — ECharts 七维雷达图
    GET    /cc3/viz/export-all                  — 一键导出全部格式

    # 9. 健康检查 (Health)
    GET    /cc3/health                          — 健康检查
    GET    /cc3/health/ready                    — 就绪检查 (含子系统状态)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .kpa_engine import KPAEngine
from .debate_logger import DebateLogger
from .provenance_chain_builder import ProvenanceChainBuilder
from .ledger_integration import LedgerIntegration
from .query_engine import QueryEngine
from .cc_integration import CCIntegration
from .metrics import KPAMetricsEngine
from .visualizer_adapter import ProvenanceVisualizer
from .models import (
    KPAAnnotation,
    TargetType,
    SourceDimension,
    GenerationDimension,
    ValidationDimension,
    DecisionDimension,
    EvolutionDimension,
    PropagationDimension,
    RelationDimension,
    SourceTier,
    LogVerbosity,
    EventType,
    CrossLayerDirection,
)
from .exceptions import (
    CC3Error,
    AnnotationNotFoundError,
    DebateLogNotFoundError,
    HashMismatchError,
    ChainBrokenError,
)

logger = logging.getLogger("dy3_polaris.l0.cc3.api")

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
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
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


def _ok_data(resp: dict[str, Any]) -> Any:
    """从成功响应中提取 data 字段."""
    return resp.get("data")


# ============================================================
# CC3APIRouter
# ============================================================


class CC3APIRouter:
    """CC3 溯源捕获 REST API 路由器.

    将 CC3 全部子系统 (KPA 标注引擎 / 辩论日志引擎 / 溯源链构建器 /
    L0 Ledger 集成 / 查询引擎 / CC1/CC2 集成 / KPI 指标 / 可视化适配器)
    暴露为统一的 RESTful JSON API。

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
        router = CC3APIRouter()

        # 2. 直接编程调用 (返回 dict)
        result = router.create_annotation({
            "target_type": "kp",
            "target_id": "kp-dy3-yag-4f",
            "source": {"primary_source": "10.1016/j.jlumin.2019.116789"},
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
        kpa_engine: KPAEngine | None = None,
        debate_logger: DebateLogger | None = None,
        chain_builder: ProvenanceChainBuilder | None = None,
        ledger: LedgerIntegration | None = None,
        query_engine: QueryEngine | None = None,
        cc_integration: CCIntegration | None = None,
        metrics_engine: KPAMetricsEngine | None = None,
        visualizer: ProvenanceVisualizer | None = None,
    ) -> None:
        """初始化 CC3 API 路由器.

        Args:
            kpa_engine: KPA 七维标注引擎 (None 则自动创建)
            debate_logger: 辩论日志引擎 (None 则自动创建)
            chain_builder: 溯源链构建器 (None 则自动创建)
            ledger: L0 Ledger 集成器 (None 则自动创建)
            query_engine: 查询引擎 (None 则自动创建)
            cc_integration: CC1/CC2 集成器 (None 则自动创建)
            metrics_engine: KPI 指标引擎 (None 则自动创建)
            visualizer: 可视化适配器 (None 则自动创建)
        """
        self._kpa = kpa_engine or KPAEngine()
        self._dl = debate_logger or DebateLogger()
        self._chain = chain_builder or ProvenanceChainBuilder()
        self._ledger = ledger or LedgerIntegration()
        self._query = query_engine or QueryEngine(
            self._kpa, self._dl, self._chain, self._ledger
        )
        self._integration = cc_integration or CCIntegration(
            self._kpa, self._dl, self._chain, self._ledger
        )
        self._metrics = metrics_engine or KPAMetricsEngine(
            self._kpa, self._dl, self._chain, self._ledger
        )
        self._visualizer = visualizer or ProvenanceVisualizer(
            self._kpa, self._dl, self._chain
        )
        self._lock = threading.RLock()
        self._started_at = time.time()

    # ==========================================================
    # 1. KPA 标注 (Annotation)
    # ==========================================================

    def create_annotation(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/annotation/create — 创建 KPA 标注.

        请求体字段 (全部可选):
            target_type, target_id, target_metadata,
            source, generation, validation, decision,
            evolution, propagation, relation, annotator_agent

        Returns:
            新创建的 KPAAnnotation
        """
        try:
            # 解析七维数据
            kwargs: dict[str, Any] = {}

            if "target_type" in body:
                kwargs["target_type"] = TargetType(body["target_type"])
            if "target_id" in body:
                kwargs["target_id"] = body["target_id"]
            if "target_metadata" in body:
                kwargs["target_metadata"] = body["target_metadata"]
            if "annotator_agent" in body:
                kwargs["annotator_agent"] = body["annotator_agent"]

            # 七维标注数据
            for dim_name, dim_cls in [
                ("source", SourceDimension),
                ("generation", GenerationDimension),
                ("validation", ValidationDimension),
                ("decision", DecisionDimension),
                ("evolution", EvolutionDimension),
                ("propagation", PropagationDimension),
                ("relation", RelationDimension),
            ]:
                if dim_name in body and isinstance(body[dim_name], dict):
                    kwargs[dim_name] = dim_cls(**body[dim_name])

            annotation = self._kpa.create_annotation(**kwargs)
            return _ok(_serialize(annotation), "标注创建成功")
        except CC3Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            logger.exception("创建标注失败")
            return _err(500, f"创建标注失败: {exc}")

    def get_annotation(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/annotation/{annotation_id} — 获取标注详情."""
        try:
            annotation = self._kpa.get_annotation(annotation_id)
            return _ok(_serialize(annotation))
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"获取标注失败: {exc}")

    def list_annotations(
        self,
        target_type: str = "",
        target_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /cc3/annotation/list — 列出标注."""
        try:
            tt = TargetType(target_type) if target_type else None
            annotations = self._kpa.list_annotations(
                target_type=tt,
                target_id=target_id or None,
                limit=limit,
                offset=offset,
            )
            return _ok({
                "items": [_serialize(a) for a in annotations],
                "total": len(annotations),
                "limit": limit,
                "offset": offset,
            })
        except Exception as exc:
            return _err(500, f"列出标注失败: {exc}")

    def update_dimension(
        self, annotation_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/annotation/{annotation_id}/dimension — 更新维度.

        请求体:
            dimension: 维度名称 (source/generation/validation/decision/evolution/propagation/relation)
            data: 维度数据字典
        """
        try:
            dimension = body.get("dimension", "")
            data = body.get("data", {})
            if not dimension:
                return _err(400, "缺少 dimension 字段")
            annotation = self._kpa.update_dimension(annotation_id, dimension, data)
            return _ok(_serialize(annotation), f"维度 {dimension} 更新成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except CC3Error as exc:
            return _err(400, str(exc))
        except Exception as exc:
            return _err(500, f"更新维度失败: {exc}")

    def update_validation(
        self, annotation_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/annotation/{annotation_id}/validation — 更新校验维度."""
        try:
            annotation = self._kpa.update_validation(
                annotation_id=annotation_id,
                cc1_review_id=body.get("cc1_review_id", ""),
                four_layer_scores=body.get("four_layer_scores", {}),
                verdict=body.get("verdict", "pass"),
                validation_issues=body.get("validation_issues", []),
                standard_value_check=body.get("standard_value_check", {}),
                mcp_tool_calls=body.get("mcp_tool_calls", []),
                self_correction_count=body.get("self_correction_count", 0),
            )
            return _ok(_serialize(annotation), "校验维度更新成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"更新校验维度失败: {exc}")

    def update_decision(
        self, annotation_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/annotation/{annotation_id}/decision — 更新决策维度."""
        try:
            annotation = self._kpa.update_decision(
                annotation_id=annotation_id,
                meta_decider_result=body.get("meta_decider_result", ""),
                paradigm_selected=body.get("paradigm_selected", ""),
                adjudicator_verdict=body.get("adjudicator_verdict", ""),
                cc2_approval_id=body.get("cc2_approval_id", ""),
                cc2_approval_level=body.get("cc2_approval_level", ""),
                debate_id=body.get("debate_id", ""),
                decision_path=body.get("decision_path", []),
            )
            return _ok(_serialize(annotation), "决策维度更新成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"更新决策维度失败: {exc}")

    def record_propagation(
        self, annotation_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/annotation/{annotation_id}/propagation — 记录传播."""
        try:
            annotation = self._kpa.record_propagation(
                annotation_id=annotation_id,
                session_id=body.get("session_id", ""),
                agent_id=body.get("agent_id", ""),
                learner_id=body.get("learner_id", ""),
                interaction_type=body.get("interaction_type", ""),
            )
            return _ok(_serialize(annotation), "传播记录成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"记录传播失败: {exc}")

    def apply_rules(self, annotation_id: str) -> dict[str, Any]:
        """POST /cc3/annotation/{annotation_id}/rules — 应用 Dy3+ 规则."""
        try:
            report = self._kpa.apply_rules(annotation_id)
            return _ok(_serialize(report), "规则应用完成")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"应用规则失败: {exc}")

    def evaluate_completeness(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/annotation/{annotation_id}/completeness — 评估完整度."""
        try:
            report = self._kpa.evaluate_completeness(annotation_id)
            return _ok(_serialize(report))
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"评估完整度失败: {exc}")

    def verify_hash(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/annotation/{annotation_id}/verify-hash — 哈希校验."""
        try:
            passed = self._kpa.verify_hash(annotation_id)
            return _ok({"passed": passed, "annotation_id": annotation_id})
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except HashMismatchError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"哈希校验失败: {exc}")

    def verify_signature(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/annotation/{annotation_id}/verify-signature — 签名校验."""
        try:
            valid = self._kpa.verify_signature(annotation_id)
            return _ok({"valid": valid, "annotation_id": annotation_id})
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"签名校验失败: {exc}")

    def to_prov_model(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/annotation/{annotation_id}/prov — W3C PROV 映射."""
        try:
            prov = self._kpa.to_prov_model(annotation_id)
            return _ok(_serialize(prov))
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"PROV 映射失败: {exc}")

    def annotation_statistics(self) -> dict[str, Any]:
        """GET /cc3/annotation/statistics — 标注统计."""
        try:
            stats = self._kpa.statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            return _err(500, f"获取统计失败: {exc}")

    # ==========================================================
    # 2. 辩论日志 (Debate Log)
    # ==========================================================

    def create_debate_log(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/debate/create — 创建辩论日志."""
        try:
            verbosity = LogVerbosity(body.get("verbosity", "summary"))
            log = self._dl.create_log(
                debate_id=body.get("debate_id", ""),
                task_id=body.get("task_id", ""),
                session_id=body.get("session_id", ""),
                trigger_reason=body.get("trigger_reason", ""),
                complexity_score=body.get("complexity_score", 0.0),
                verbosity=verbosity,
                max_rounds=body.get("max_rounds", 3),
                convergence_threshold=body.get("convergence_threshold", 0.1),
            )
            return _ok(_serialize(log), "辩论日志创建成功")
        except Exception as exc:
            return _err(500, f"创建辩论日志失败: {exc}")

    def add_debate_round(
        self, log_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/round — 追加辩论轮次."""
        try:
            from .models import DebateArgument, DebateCounterargument

            generator_args = [
                DebateArgument(**a) if isinstance(a, dict) else a
                for a in body.get("generator_arguments", [])
            ]
            reviewer_counters = [
                DebateCounterargument(**c) if isinstance(c, dict) else c
                for c in body.get("reviewer_counterarguments", [])
            ]
            log = self._dl.add_round(
                log_id,
                generator_args,
                reviewer_counters,
                round_duration_ms=body.get("round_duration_ms", 0.0),
            )
            return _ok(_serialize(log), "轮次追加成功")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"追加轮次失败: {exc}")

    def check_convergence(self, log_id: str) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/convergence — 检查收敛."""
        try:
            result = self._dl.check_convergence(log_id)
            return _ok(_serialize(result))
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"收敛检查失败: {exc}")

    def record_adjudication(
        self, log_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/adjudication — 记录裁决."""
        try:
            from .models import AdjudicatorVerdict

            verdict_data = body.get("verdict", {})
            if isinstance(verdict_data, dict):
                verdict = AdjudicatorVerdict(**verdict_data)
            else:
                verdict = verdict_data
            log = self._dl.record_adjudication(log_id, verdict)
            return _ok(_serialize(log), "裁决记录成功")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"记录裁决失败: {exc}")

    def record_outcome(
        self, log_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/outcome — 记录结果."""
        try:
            log = self._dl.record_outcome(
                log_id,
                final_content=body.get("final_content", ""),
                affected_annotations=body.get("affected_annotations", []),
                knowledge_updates=body.get("knowledge_updates", []),
                consensus_reached=body.get("consensus_reached", False),
            )
            return _ok(_serialize(log), "结果记录成功")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"记录结果失败: {exc}")

    def record_resource_usage(
        self, log_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/resource — 记录资源消耗."""
        try:
            log = self._dl.record_resource_usage(
                log_id,
                total_tokens=body.get("total_tokens", 0),
                total_cost_usd=body.get("total_cost_usd", 0.0),
                total_duration_ms=body.get("total_duration_ms", 0.0),
                api_calls=body.get("api_calls", 0),
            )
            return _ok(_serialize(log), "资源消耗记录成功")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"记录资源消耗失败: {exc}")

    def finalize_debate_log(self, log_id: str) -> dict[str, Any]:
        """POST /cc3/debate/{log_id}/finalize — 完成日志."""
        try:
            log = self._dl.finalize(log_id)
            return _ok(_serialize(log), "日志已完成")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"完成日志失败: {exc}")

    def verify_debate_integrity(self, log_id: str) -> dict[str, Any]:
        """GET /cc3/debate/{log_id}/verify — 完整性校验."""
        try:
            result = self._dl.verify_integrity(log_id)
            return _ok(_serialize(result))
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"完整性校验失败: {exc}")

    def export_debate_log(
        self, log_id: str, verbosity: str = "summary"
    ) -> dict[str, Any]:
        """GET /cc3/debate/{log_id}/export — 导出日志."""
        try:
            v = LogVerbosity(verbosity)
            exported = self._dl.export_log(log_id, v)
            return _ok(_serialize(exported))
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"导出日志失败: {exc}")

    def get_debate_log(self, log_id: str) -> dict[str, Any]:
        """GET /cc3/debate/{log_id} — 获取日志."""
        try:
            log = self._dl.get_log(log_id)
            return _ok(_serialize(log))
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"获取日志失败: {exc}")

    def list_debate_logs(
        self,
        task_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /cc3/debate/list — 列出日志."""
        try:
            logs = self._dl.list_logs(
                task_id=task_id or None,
                limit=limit,
                offset=offset,
            )
            return _ok({
                "items": [_serialize(l) for l in logs],
                "total": len(logs),
                "limit": limit,
                "offset": offset,
            })
        except Exception as exc:
            return _err(500, f"列出日志失败: {exc}")

    def debate_statistics(self) -> dict[str, Any]:
        """GET /cc3/debate/statistics — 辩论统计."""
        try:
            stats = self._dl.statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            return _err(500, f"获取统计失败: {exc}")

    # ==========================================================
    # 3. 溯源链 (Chain)
    # ==========================================================

    def create_chain(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/chain/create — 创建溯源链."""
        try:
            chain_id = self._chain.create_chain(
                chain_id=body.get("chain_id", ""),
                metadata=body.get("metadata", {}),
            )
            return _ok({"chain_id": chain_id}, "溯源链创建成功")
        except Exception as exc:
            return _err(500, f"创建溯源链失败: {exc}")

    def append_chain_node(
        self, chain_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /cc3/chain/{chain_id}/append — 追加节点."""
        try:
            cross_layer = None
            if body.get("cross_layer_direction"):
                cross_layer = CrossLayerDirection(body["cross_layer_direction"])
            node = self._chain.append_node(
                chain_id=chain_id,
                annotation_id=body.get("annotation_id", ""),
                agent_id=body.get("agent_id", ""),
                action=body.get("action", ""),
                layer=body.get("layer", ""),
                cross_layer_direction=cross_layer,
                metadata=body.get("metadata", {}),
            )
            return _ok(_serialize(node), "节点追加成功")
        except ChainBrokenError as exc:
            return _err(409, str(exc))
        except Exception as exc:
            return _err(500, f"追加节点失败: {exc}")

    def verify_chain(self, chain_id: str) -> dict[str, Any]:
        """GET /cc3/chain/{chain_id}/verify — 验证链完整性."""
        try:
            report = self._chain.verify_chain(chain_id)
            return _ok(_serialize(report))
        except Exception as exc:
            return _err(500, f"验证链失败: {exc}")

    def verify_chain_node(
        self, chain_id: str, node_index: int = 0
    ) -> dict[str, Any]:
        """GET /cc3/chain/{chain_id}/verify-node — 验证单个节点."""
        try:
            passed = self._chain.verify_node(chain_id, node_index)
            return _ok({"passed": passed, "node_index": node_index})
        except Exception as exc:
            return _err(500, f"验证节点失败: {exc}")

    def build_merkle_tree(self, chain_id: str) -> dict[str, Any]:
        """POST /cc3/chain/{chain_id}/merkle — 构建 Merkle 树."""
        try:
            root = self._chain.build_merkle_tree(chain_id)
            return _ok({"merkle_root": root, "chain_id": chain_id}, "Merkle 树构建成功")
        except Exception as exc:
            return _err(500, f"构建 Merkle 树失败: {exc}")

    def get_merkle_proof(
        self, chain_id: str, node_index: int = 0
    ) -> dict[str, Any]:
        """GET /cc3/chain/{chain_id}/merkle-proof — 获取 Merkle 证明."""
        try:
            proof = self._chain.get_merkle_proof(chain_id, node_index)
            return _ok({
                "proof": _serialize(proof),
                "node_index": node_index,
                "chain_id": chain_id,
            })
        except Exception as exc:
            return _err(500, f"获取 Merkle 证明失败: {exc}")

    def compress_chain(self, chain_id: str) -> dict[str, Any]:
        """POST /cc3/chain/{chain_id}/compress — 压缩链."""
        try:
            result = self._chain.compress(chain_id)
            return _ok(_serialize(result), "链压缩成功")
        except Exception as exc:
            return _err(500, f"压缩链失败: {exc}")

    def snapshot_chain(self, chain_id: str) -> dict[str, Any]:
        """GET /cc3/chain/{chain_id}/snapshot — 创建快照."""
        try:
            snapshot = self._chain.snapshot(chain_id)
            return _ok(_serialize(snapshot))
        except Exception as exc:
            return _err(500, f"创建快照失败: {exc}")

    def audit_verify(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/chain/audit-verify — 审计验证."""
        try:
            result = self._chain.audit_verify(
                chain_id=body.get("chain_id", ""),
                expected_root=body.get("expected_root", ""),
                node_index=body.get("node_index", 0),
                proof=body.get("proof", []),
            )
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"审计验证失败: {exc}")

    def list_chains(self) -> dict[str, Any]:
        """GET /cc3/chain/list — 列出溯源链."""
        try:
            chains = self._chain.list_chains()
            return _ok({"items": _serialize(chains), "total": len(chains)})
        except Exception as exc:
            return _err(500, f"列出溯源链失败: {exc}")

    def trace_cross_layer(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/chain/trace-cross-layer — 跨层追踪."""
        try:
            direction = CrossLayerDirection(body.get("direction", "l2_to_l3"))
            result = self._chain.trace_cross_layer(
                trace_id=body.get("trace_id", ""),
                direction=direction,
            )
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"跨层追踪失败: {exc}")

    # ==========================================================
    # 4. L0 Ledger (Ledger)
    # ==========================================================

    def ledger_write_kpa(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/ledger/write-kpa — 写入 KPA 事件."""
        try:
            annotation_id = body.get("annotation_id", "")
            annotation = self._kpa.get_annotation(annotation_id)
            event = self._ledger.write_kpa(
                annotation,
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
            )
            return _ok(_serialize(event), "KPA 事件写入成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"写入 KPA 事件失败: {exc}")

    def ledger_write_dl(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/ledger/write-dl — 写入 DL 事件."""
        try:
            log_id = body.get("debate_log_id", "")
            log = self._dl.get_log(log_id)
            event = self._ledger.write_dl(
                log,
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
            )
            return _ok(_serialize(event), "DL 事件写入成功")
        except DebateLogNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"写入 DL 事件失败: {exc}")

    def ledger_write_cross_layer(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/ledger/write-cross-layer — 写入跨层事件."""
        try:
            direction = CrossLayerDirection(body.get("direction", "l2_to_l3"))
            event = self._ledger.write_cross_layer(
                direction=direction,
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
                payload=body.get("payload", {}),
            )
            return _ok(_serialize(event), "跨层事件写入成功")
        except Exception as exc:
            return _err(500, f"写入跨层事件失败: {exc}")

    def ledger_write_human_override(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/ledger/write-human-override — 写入人工干预."""
        try:
            event = self._ledger.write_human_override(
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
                user_id=body.get("user_id", ""),
                override_type=body.get("override_type", ""),
                target_id=body.get("target_id", ""),
                reason=body.get("reason", ""),
                payload=body.get("payload", {}),
            )
            return _ok(_serialize(event), "人工干预事件写入成功")
        except Exception as exc:
            return _err(500, f"写入人工干预事件失败: {exc}")

    def ledger_get_event(self, event_id: str) -> dict[str, Any]:
        """GET /cc3/ledger/event/{event_id} — 获取事件."""
        try:
            event = self._ledger.get_event(event_id)
            if event is None:
                return _err(404, f"事件未找到: {event_id}")
            return _ok(_serialize(event))
        except Exception as exc:
            return _err(500, f"获取事件失败: {exc}")

    def ledger_query(
        self,
        trace_id: str = "",
        event_type: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """GET /cc3/ledger/query — 查询事件."""
        try:
            et = EventType(event_type) if event_type else None
            events = self._ledger.query(
                trace_id=trace_id or None,
                event_type=et,
                limit=limit,
            )
            return _ok({
                "items": [_serialize(e) for e in events],
                "total": len(events),
            })
        except Exception as exc:
            return _err(500, f"查询事件失败: {exc}")

    def ledger_query_by_time(
        self,
        start_time: float = 0.0,
        end_time: float = 0.0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """GET /cc3/ledger/query-by-time — 按时间范围查询."""
        try:
            if end_time == 0.0:
                end_time = time.time()
            if start_time == 0.0:
                start_time = 0.0
            events = self._ledger.query_by_time_range(start_time, end_time, limit)
            return _ok({
                "items": [_serialize(e) for e in events],
                "total": len(events),
                "start_time": start_time,
                "end_time": end_time,
            })
        except Exception as exc:
            return _err(500, f"按时间范围查询失败: {exc}")

    def ledger_verify(self) -> dict[str, Any]:
        """GET /cc3/ledger/verify — 验证 Ledger."""
        try:
            result = self._ledger.verify_ledger()
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"验证 Ledger 失败: {exc}")

    def ledger_statistics(self) -> dict[str, Any]:
        """GET /cc3/ledger/statistics — Ledger 统计."""
        try:
            stats = self._ledger.statistics()
            return _ok(_serialize(stats))
        except Exception as exc:
            return _err(500, f"获取统计失败: {exc}")

    # ==========================================================
    # 5. 查询引擎 (Query)
    # ==========================================================

    def query_trace(self, trace_id: str) -> dict[str, Any]:
        """GET /cc3/query/trace/{trace_id} — 按 trace_id 回溯."""
        try:
            result = self._query.trace_by_trace_id(trace_id)
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"trace 回溯失败: {exc}")

    def query_knowledge_provenance(self, target_id: str) -> dict[str, Any]:
        """GET /cc3/query/knowledge/{target_id} — 知识溯源档案."""
        try:
            result = self._query.get_knowledge_provenance(target_id)
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"知识溯源档案查询失败: {exc}")

    def query_agent_history(self, agent_id: str) -> dict[str, Any]:
        """GET /cc3/query/agent/{agent_id} — Agent 操作历史."""
        try:
            result = self._query.get_agent_history(agent_id)
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"Agent 历史查询失败: {exc}")

    def query_timeline(
        self,
        start_time: float = 0.0,
        end_time: float = 0.0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """GET /cc3/query/timeline — 时间线查询."""
        try:
            if end_time == 0.0:
                end_time = time.time()
            result = self._query.get_timeline(start_time, end_time, limit)
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"时间线查询失败: {exc}")

    def query_graph(self, target_id: str = "") -> dict[str, Any]:
        """GET /cc3/query/graph — 溯源图."""
        try:
            result = self._query.get_provenance_graph(target_id or None)
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"溯源图查询失败: {exc}")

    def query_overview(self) -> dict[str, Any]:
        """GET /cc3/query/overview — 全局概览."""
        try:
            result = self._query.overview()
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"全局概览失败: {exc}")

    # ==========================================================
    # 6. CC1/CC2 集成 (Integration)
    # ==========================================================

    def integration_cc1_review(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/integration/cc1-review — CC1 评审回调."""
        try:
            result = self._integration.on_cc1_review_completed(
                annotation_id=body.get("annotation_id", ""),
                review_id=body.get("review_id", ""),
                scores=body.get("scores", {}),
                verdict=body.get("verdict", "pass"),
                issues=body.get("issues"),
                self_correction_count=body.get("self_correction_count", 0),
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
            )
            return _ok(_serialize(result), "CC1 评审回调处理成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"CC1 评审回调失败: {exc}")

    def integration_cc2_approval(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/integration/cc2-approval — CC2 审批回调."""
        try:
            result = self._integration.on_cc2_approval_completed(
                annotation_id=body.get("annotation_id", ""),
                approval_id=body.get("approval_id", ""),
                approval_level=body.get("approval_level", ""),
                trace_id=body.get("trace_id", ""),
                session_id=body.get("session_id", ""),
            )
            return _ok(_serialize(result), "CC2 审批回调处理成功")
        except AnnotationNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:
            return _err(500, f"CC2 审批回调失败: {exc}")

    def integration_check_provenance(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/integration/check-provenance — 溯源检查 (CC1)."""
        try:
            result = self._integration.check_provenance_for_cc1(
                annotation_id=body.get("annotation_id", ""),
                target_type=body.get("target_type", "kp"),
            )
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"溯源检查失败: {exc}")

    def integration_check_escalation(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/integration/check-escalation — 升级检查 (CC2)."""
        try:
            result = self._integration.check_escalation_for_cc2(
                annotation_id=body.get("annotation_id", ""),
                operation_type=body.get("operation_type", ""),
            )
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"升级检查失败: {exc}")

    def integration_check_debate(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /cc3/integration/check-debate — 辩论触发检查."""
        try:
            result = self._integration.check_debate_trigger(
                annotation_id=body.get("annotation_id", ""),
                complexity_score=body.get("complexity_score", 0.0),
            )
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"辩论触发检查失败: {exc}")

    # ==========================================================
    # 7. KPI 指标 (Metrics)
    # ==========================================================

    def metrics_dashboard(self) -> dict[str, Any]:
        """GET /cc3/metrics/dashboard — KPI 仪表盘."""
        try:
            result = self._metrics.collect_all()
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"获取仪表盘失败: {exc}")

    def metrics_collect_all(self) -> dict[str, Any]:
        """GET /cc3/metrics/collect-all — 全量采集."""
        try:
            result = self._metrics.collect_all()
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"全量采集失败: {exc}")

    def metrics_coverage(self) -> dict[str, Any]:
        """GET /cc3/metrics/coverage — 覆盖率采集."""
        try:
            samples = self._metrics.collect_coverage()
            return _ok({"samples": [_serialize(s) for s in samples]})
        except Exception as exc:
            return _err(500, f"覆盖率采集失败: {exc}")

    def metrics_integrity(self) -> dict[str, Any]:
        """GET /cc3/metrics/integrity — 完整性采集."""
        try:
            samples = self._metrics.collect_integrity()
            return _ok({"samples": [_serialize(s) for s in samples]})
        except Exception as exc:
            return _err(500, f"完整性采集失败: {exc}")

    def metrics_performance(self) -> dict[str, Any]:
        """GET /cc3/metrics/performance — 性能采集."""
        try:
            samples = self._metrics.collect_performance()
            return _ok({"samples": [_serialize(s) for s in samples]})
        except Exception as exc:
            return _err(500, f"性能采集失败: {exc}")

    def metrics_compliance(self) -> dict[str, Any]:
        """GET /cc3/metrics/compliance — 合规性采集."""
        try:
            samples = self._metrics.collect_compliance()
            return _ok({"samples": [_serialize(s) for s in samples]})
        except Exception as exc:
            return _err(500, f"合规性采集失败: {exc}")

    def metrics_export_dashboard(self) -> dict[str, Any]:
        """GET /cc3/metrics/export-dashboard — 导出仪表盘."""
        try:
            result = self._metrics.export_dashboard()
            return _ok(_serialize(result))
        except Exception as exc:
            return _err(500, f"导出仪表盘失败: {exc}")

    # ==========================================================
    # 8. 可视化 (Visualization)
    # ==========================================================

    def viz_cytoscape(self, target_id: str = "") -> dict[str, Any]:
        """GET /cc3/viz/cytoscape — Cytoscape.js 图数据."""
        try:
            data = self._visualizer.to_cytoscape(target_id or None)
            return _ok(_serialize(data))
        except Exception as exc:
            return _err(500, f"生成 Cytoscape 数据失败: {exc}")

    def viz_d3_hierarchy(self, chain_id: str) -> dict[str, Any]:
        """GET /cc3/viz/d3-hierarchy — D3.js 层级树."""
        try:
            data = self._visualizer.to_d3_hierarchy(chain_id)
            return _ok(_serialize(data))
        except Exception as exc:
            return _err(500, f"生成 D3 层级树失败: {exc}")

    def viz_mermaid(
        self, chain_id: str = "", annotation_id: str = ""
    ) -> dict[str, Any]:
        """GET /cc3/viz/mermaid — Mermaid 流程图."""
        try:
            text = self._visualizer.to_mermaid(
                chain_id=chain_id or None,
                annotation_id=annotation_id or None,
            )
            return _ok({"mermaid_text": text})
        except Exception as exc:
            return _err(500, f"生成 Mermaid 图失败: {exc}")

    def viz_echarts_timeline(self, debate_log_id: str = "") -> dict[str, Any]:
        """GET /cc3/viz/echarts-timeline — ECharts 辩论时间线."""
        try:
            data = self._visualizer.to_echarts_timeline(debate_log_id or None)
            return _ok(_serialize(data))
        except Exception as exc:
            return _err(500, f"生成 ECharts 时间线失败: {exc}")

    def viz_echarts_radar(self, annotation_id: str) -> dict[str, Any]:
        """GET /cc3/viz/echarts-radar — ECharts 七维雷达图."""
        try:
            data = self._visualizer.to_echarts_radar(annotation_id)
            return _ok(_serialize(data))
        except Exception as exc:
            return _err(500, f"生成 ECharts 雷达图失败: {exc}")

    def viz_export_all(self, target_id: str = "") -> dict[str, Any]:
        """GET /cc3/viz/export-all — 一键导出全部格式."""
        try:
            data = self._visualizer.export_all(target_id or None)
            return _ok(_serialize(data))
        except Exception as exc:
            return _err(500, f"导出可视化数据失败: {exc}")

    # ==========================================================
    # 9. 健康检查 (Health)
    # ==========================================================

    def health(self) -> dict[str, Any]:
        """GET /cc3/health — 健康检查.

        Returns:
            健康状态字典
        """
        return _ok({
            "status": "healthy",
            "service": "cc3-provenance-capture",
            "uptime_seconds": round(time.time() - self._started_at, 3),
        })

    def health_ready(self) -> dict[str, Any]:
        """GET /cc3/health/ready — 就绪检查 (含子系统状态).

        检查全部子系统的就绪状态:
        - kpa_engine: KPA 标注引擎
        - debate_logger: 辩论日志引擎
        - chain_builder: 溯源链构建器
        - ledger: L0 Ledger 集成
        - query_engine: 查询引擎
        - cc_integration: CC1/CC2 集成
        - metrics_engine: KPI 指标引擎
        - visualizer: 可视化适配器

        Returns:
            就绪状态及各子系统状态
        """
        subsystems: dict[str, dict[str, Any]] = {}
        all_ready = True

        # KPA 引擎
        try:
            stats = self._kpa.statistics()
            subsystems["kpa_engine"] = {
                "ready": True,
                "total_annotations": stats.get("total_annotations", 0),
            }
        except Exception as exc:
            subsystems["kpa_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 辩论日志引擎
        try:
            dl_stats = self._dl.statistics()
            subsystems["debate_logger"] = {
                "ready": True,
                "total_logs": dl_stats.get("total_logs", 0),
            }
        except Exception as exc:
            subsystems["debate_logger"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 溯源链构建器
        try:
            chains = self._chain.list_chains()
            subsystems["chain_builder"] = {
                "ready": True,
                "total_chains": len(chains),
            }
        except Exception as exc:
            subsystems["chain_builder"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # L0 Ledger
        try:
            ledger_stats = self._ledger.statistics()
            subsystems["ledger"] = {
                "ready": True,
                "total_events": ledger_stats.get("total_events", 0),
            }
        except Exception as exc:
            subsystems["ledger"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 查询引擎
        try:
            overview = self._query.overview()
            subsystems["query_engine"] = {
                "ready": True,
                "overview_keys": len(overview),
            }
        except Exception as exc:
            subsystems["query_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # CC1/CC2 集成
        subsystems["cc_integration"] = {"ready": True}

        # KPI 指标引擎
        try:
            dashboard = self._metrics.export_dashboard()
            subsystems["metrics_engine"] = {
                "ready": True,
                "categories": len(dashboard.get("categories", {})),
            }
        except Exception as exc:
            subsystems["metrics_engine"] = {"ready": False, "error": str(exc)}
            all_ready = False

        # 可视化适配器
        subsystems["visualizer"] = {"ready": True}

        return _ok({
            "ready": all_ready,
            "subsystems": subsystems,
            "uptime_seconds": round(time.time() - self._started_at, 3),
        })

    # ==========================================================
    # 路由表 & Starlette 应用
    # ==========================================================

    def get_routes(self) -> list[dict[str, Any]]:
        """返回全部 API 端点的路由表.

        Returns:
            路由列表, 每项包含 path, method, description
        """
        return [
            # 1. KPA 标注
            {"path": "/cc3/annotation/create", "method": "POST", "description": "创建 KPA 标注"},
            {"path": "/cc3/annotation/{annotation_id}", "method": "GET", "description": "获取标注详情"},
            {"path": "/cc3/annotation/list", "method": "GET", "description": "列出标注"},
            {"path": "/cc3/annotation/{annotation_id}/dimension", "method": "POST", "description": "更新维度"},
            {"path": "/cc3/annotation/{annotation_id}/validation", "method": "POST", "description": "更新校验维度"},
            {"path": "/cc3/annotation/{annotation_id}/decision", "method": "POST", "description": "更新决策维度"},
            {"path": "/cc3/annotation/{annotation_id}/propagation", "method": "POST", "description": "记录传播"},
            {"path": "/cc3/annotation/{annotation_id}/rules", "method": "POST", "description": "应用 Dy3+ 规则"},
            {"path": "/cc3/annotation/{annotation_id}/completeness", "method": "GET", "description": "评估完整度"},
            {"path": "/cc3/annotation/{annotation_id}/verify-hash", "method": "GET", "description": "哈希校验"},
            {"path": "/cc3/annotation/{annotation_id}/verify-signature", "method": "GET", "description": "签名校验"},
            {"path": "/cc3/annotation/{annotation_id}/prov", "method": "GET", "description": "W3C PROV 映射"},
            {"path": "/cc3/annotation/statistics", "method": "GET", "description": "标注统计"},
            # 2. 辩论日志
            {"path": "/cc3/debate/create", "method": "POST", "description": "创建辩论日志"},
            {"path": "/cc3/debate/{log_id}/round", "method": "POST", "description": "追加辩论轮次"},
            {"path": "/cc3/debate/{log_id}/convergence", "method": "POST", "description": "检查收敛"},
            {"path": "/cc3/debate/{log_id}/adjudication", "method": "POST", "description": "记录裁决"},
            {"path": "/cc3/debate/{log_id}/outcome", "method": "POST", "description": "记录结果"},
            {"path": "/cc3/debate/{log_id}/resource", "method": "POST", "description": "记录资源消耗"},
            {"path": "/cc3/debate/{log_id}/finalize", "method": "POST", "description": "完成日志"},
            {"path": "/cc3/debate/{log_id}/verify", "method": "GET", "description": "完整性校验"},
            {"path": "/cc3/debate/{log_id}/export", "method": "GET", "description": "导出日志"},
            {"path": "/cc3/debate/{log_id}", "method": "GET", "description": "获取日志"},
            {"path": "/cc3/debate/list", "method": "GET", "description": "列出日志"},
            {"path": "/cc3/debate/statistics", "method": "GET", "description": "辩论统计"},
            # 3. 溯源链
            {"path": "/cc3/chain/create", "method": "POST", "description": "创建溯源链"},
            {"path": "/cc3/chain/{chain_id}/append", "method": "POST", "description": "追加节点"},
            {"path": "/cc3/chain/{chain_id}/verify", "method": "GET", "description": "验证链完整性"},
            {"path": "/cc3/chain/{chain_id}/verify-node", "method": "GET", "description": "验证单个节点"},
            {"path": "/cc3/chain/{chain_id}/merkle", "method": "POST", "description": "构建 Merkle 树"},
            {"path": "/cc3/chain/{chain_id}/merkle-proof", "method": "GET", "description": "获取 Merkle 证明"},
            {"path": "/cc3/chain/{chain_id}/compress", "method": "POST", "description": "压缩链"},
            {"path": "/cc3/chain/{chain_id}/snapshot", "method": "GET", "description": "创建快照"},
            {"path": "/cc3/chain/audit-verify", "method": "POST", "description": "审计验证"},
            {"path": "/cc3/chain/list", "method": "GET", "description": "列出溯源链"},
            {"path": "/cc3/chain/trace-cross-layer", "method": "POST", "description": "跨层追踪"},
            # 4. L0 Ledger
            {"path": "/cc3/ledger/write-kpa", "method": "POST", "description": "写入 KPA 事件"},
            {"path": "/cc3/ledger/write-dl", "method": "POST", "description": "写入 DL 事件"},
            {"path": "/cc3/ledger/write-cross-layer", "method": "POST", "description": "写入跨层事件"},
            {"path": "/cc3/ledger/write-human-override", "method": "POST", "description": "写入人工干预"},
            {"path": "/cc3/ledger/event/{event_id}", "method": "GET", "description": "获取事件"},
            {"path": "/cc3/ledger/query", "method": "GET", "description": "查询事件"},
            {"path": "/cc3/ledger/query-by-time", "method": "GET", "description": "按时间范围查询"},
            {"path": "/cc3/ledger/verify", "method": "GET", "description": "验证 Ledger"},
            {"path": "/cc3/ledger/statistics", "method": "GET", "description": "Ledger 统计"},
            # 5. 查询引擎
            {"path": "/cc3/query/trace/{trace_id}", "method": "GET", "description": "按 trace_id 回溯"},
            {"path": "/cc3/query/knowledge/{target_id}", "method": "GET", "description": "知识溯源档案"},
            {"path": "/cc3/query/agent/{agent_id}", "method": "GET", "description": "Agent 操作历史"},
            {"path": "/cc3/query/timeline", "method": "GET", "description": "时间线查询"},
            {"path": "/cc3/query/graph", "method": "GET", "description": "溯源图"},
            {"path": "/cc3/query/overview", "method": "GET", "description": "全局概览"},
            # 6. CC1/CC2 集成
            {"path": "/cc3/integration/cc1-review", "method": "POST", "description": "CC1 评审回调"},
            {"path": "/cc3/integration/cc2-approval", "method": "POST", "description": "CC2 审批回调"},
            {"path": "/cc3/integration/check-provenance", "method": "POST", "description": "溯源检查 (CC1)"},
            {"path": "/cc3/integration/check-escalation", "method": "POST", "description": "升级检查 (CC2)"},
            {"path": "/cc3/integration/check-debate", "method": "POST", "description": "辩论触发检查"},
            # 7. KPI 指标
            {"path": "/cc3/metrics/dashboard", "method": "GET", "description": "KPI 仪表盘"},
            {"path": "/cc3/metrics/collect-all", "method": "GET", "description": "全量采集"},
            {"path": "/cc3/metrics/coverage", "method": "GET", "description": "覆盖率采集"},
            {"path": "/cc3/metrics/integrity", "method": "GET", "description": "完整性采集"},
            {"path": "/cc3/metrics/performance", "method": "GET", "description": "性能采集"},
            {"path": "/cc3/metrics/compliance", "method": "GET", "description": "合规性采集"},
            {"path": "/cc3/metrics/export-dashboard", "method": "GET", "description": "导出仪表盘"},
            # 8. 可视化
            {"path": "/cc3/viz/cytoscape", "method": "GET", "description": "Cytoscape.js 图数据"},
            {"path": "/cc3/viz/d3-hierarchy", "method": "GET", "description": "D3.js 层级树"},
            {"path": "/cc3/viz/mermaid", "method": "GET", "description": "Mermaid 流程图"},
            {"path": "/cc3/viz/echarts-timeline", "method": "GET", "description": "ECharts 辩论时间线"},
            {"path": "/cc3/viz/echarts-radar", "method": "GET", "description": "ECharts 七维雷达图"},
            {"path": "/cc3/viz/export-all", "method": "GET", "description": "一键导出全部格式"},
            # 9. 健康检查
            {"path": "/cc3/health", "method": "GET", "description": "健康检查"},
            {"path": "/cc3/health/ready", "method": "GET", "description": "就绪检查 (含子系统状态)"},
        ]

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

        # --- 路由定义 ---

        # 1. KPA 标注
        async def _annotation_create(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.create_annotation(body))

        async def _annotation_get(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.get_annotation(aid))

        async def _annotation_list(request: "Request") -> "JSONResponse":
            return _json_response(self.list_annotations(
                target_type=_qp(request, "target_type"),
                target_id=_qp(request, "target_id"),
                limit=int(_qp(request, "limit", "100")),
                offset=int(_qp(request, "offset", "0")),
            ))

        async def _annotation_dimension(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            body = await _json_body(request)
            return _json_response(self.update_dimension(aid, body))

        async def _annotation_validation(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            body = await _json_body(request)
            return _json_response(self.update_validation(aid, body))

        async def _annotation_decision(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            body = await _json_body(request)
            return _json_response(self.update_decision(aid, body))

        async def _annotation_propagation(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            body = await _json_body(request)
            return _json_response(self.record_propagation(aid, body))

        async def _annotation_rules(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.apply_rules(aid))

        async def _annotation_completeness(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.evaluate_completeness(aid))

        async def _annotation_verify_hash(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.verify_hash(aid))

        async def _annotation_verify_sig(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.verify_signature(aid))

        async def _annotation_prov(request: "Request") -> "JSONResponse":
            aid = request.path_params["annotation_id"]
            return _json_response(self.to_prov_model(aid))

        async def _annotation_stats(request: "Request") -> "JSONResponse":
            return _json_response(self.annotation_statistics())

        # 2. 辩论日志
        async def _debate_create(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.create_debate_log(body))

        async def _debate_round(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            body = await _json_body(request)
            return _json_response(self.add_debate_round(lid, body))

        async def _debate_convergence(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            return _json_response(self.check_convergence(lid))

        async def _debate_adjudication(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            body = await _json_body(request)
            return _json_response(self.record_adjudication(lid, body))

        async def _debate_outcome(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            body = await _json_body(request)
            return _json_response(self.record_outcome(lid, body))

        async def _debate_resource(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            body = await _json_body(request)
            return _json_response(self.record_resource_usage(lid, body))

        async def _debate_finalize(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            return _json_response(self.finalize_debate_log(lid))

        async def _debate_verify(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            return _json_response(self.verify_debate_integrity(lid))

        async def _debate_export(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            v = _qp(request, "verbosity", "summary")
            return _json_response(self.export_debate_log(lid, v))

        async def _debate_get(request: "Request") -> "JSONResponse":
            lid = request.path_params["log_id"]
            return _json_response(self.get_debate_log(lid))

        async def _debate_list(request: "Request") -> "JSONResponse":
            return _json_response(self.list_debate_logs(
                task_id=_qp(request, "task_id"),
                limit=int(_qp(request, "limit", "100")),
                offset=int(_qp(request, "offset", "0")),
            ))

        async def _debate_stats(request: "Request") -> "JSONResponse":
            return _json_response(self.debate_statistics())

        # 3. 溯源链
        async def _chain_create(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.create_chain(body))

        async def _chain_append(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            body = await _json_body(request)
            return _json_response(self.append_chain_node(cid, body))

        async def _chain_verify(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            return _json_response(self.verify_chain(cid))

        async def _chain_verify_node(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            idx = int(_qp(request, "node_index", "0"))
            return _json_response(self.verify_chain_node(cid, idx))

        async def _chain_merkle(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            return _json_response(self.build_merkle_tree(cid))

        async def _chain_merkle_proof(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            idx = int(_qp(request, "node_index", "0"))
            return _json_response(self.get_merkle_proof(cid, idx))

        async def _chain_compress(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            return _json_response(self.compress_chain(cid))

        async def _chain_snapshot(request: "Request") -> "JSONResponse":
            cid = request.path_params["chain_id"]
            return _json_response(self.snapshot_chain(cid))

        async def _chain_audit(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.audit_verify(body))

        async def _chain_list(request: "Request") -> "JSONResponse":
            return _json_response(self.list_chains())

        async def _chain_trace(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.trace_cross_layer(body))

        # 4. L0 Ledger
        async def _ledger_write_kpa(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.ledger_write_kpa(body))

        async def _ledger_write_dl(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.ledger_write_dl(body))

        async def _ledger_write_cl(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.ledger_write_cross_layer(body))

        async def _ledger_write_ho(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.ledger_write_human_override(body))

        async def _ledger_get_event(request: "Request") -> "JSONResponse":
            eid = request.path_params["event_id"]
            return _json_response(self.ledger_get_event(eid))

        async def _ledger_query(request: "Request") -> "JSONResponse":
            return _json_response(self.ledger_query(
                trace_id=_qp(request, "trace_id"),
                event_type=_qp(request, "event_type"),
                limit=int(_qp(request, "limit", "100")),
            ))

        async def _ledger_query_time(request: "Request") -> "JSONResponse":
            return _json_response(self.ledger_query_by_time(
                start_time=float(_qp(request, "start_time", "0")),
                end_time=float(_qp(request, "end_time", "0")),
                limit=int(_qp(request, "limit", "100")),
            ))

        async def _ledger_verify(request: "Request") -> "JSONResponse":
            return _json_response(self.ledger_verify())

        async def _ledger_stats(request: "Request") -> "JSONResponse":
            return _json_response(self.ledger_statistics())

        # 5. 查询引擎
        async def _query_trace(request: "Request") -> "JSONResponse":
            tid = request.path_params["trace_id"]
            return _json_response(self.query_trace(tid))

        async def _query_knowledge(request: "Request") -> "JSONResponse":
            tid = request.path_params["target_id"]
            return _json_response(self.query_knowledge_provenance(tid))

        async def _query_agent(request: "Request") -> "JSONResponse":
            aid = request.path_params["agent_id"]
            return _json_response(self.query_agent_history(aid))

        async def _query_timeline(request: "Request") -> "JSONResponse":
            return _json_response(self.query_timeline(
                start_time=float(_qp(request, "start_time", "0")),
                end_time=float(_qp(request, "end_time", "0")),
                limit=int(_qp(request, "limit", "100")),
            ))

        async def _query_graph(request: "Request") -> "JSONResponse":
            return _json_response(self.query_graph(_qp(request, "target_id")))

        async def _query_overview(request: "Request") -> "JSONResponse":
            return _json_response(self.query_overview())

        # 6. CC1/CC2 集成
        async def _integ_cc1(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.integration_cc1_review(body))

        async def _integ_cc2(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.integration_cc2_approval(body))

        async def _integ_prov(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.integration_check_provenance(body))

        async def _integ_esc(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.integration_check_escalation(body))

        async def _integ_deb(request: "Request") -> "JSONResponse":
            body = await _json_body(request)
            return _json_response(self.integration_check_debate(body))

        # 7. KPI 指标
        async def _metrics_dashboard(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_dashboard())

        async def _metrics_all(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_collect_all())

        async def _metrics_coverage(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_coverage())

        async def _metrics_integrity(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_integrity())

        async def _metrics_perf(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_performance())

        async def _metrics_comp(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_compliance())

        async def _metrics_export(request: "Request") -> "JSONResponse":
            return _json_response(self.metrics_export_dashboard())

        # 8. 可视化
        async def _viz_cyto(request: "Request") -> "JSONResponse":
            return _json_response(self.viz_cytoscape(_qp(request, "target_id")))

        async def _viz_d3(request: "Request") -> "JSONResponse":
            cid = _qp(request, "chain_id")
            return _json_response(self.viz_d3_hierarchy(cid))

        async def _viz_mermaid(request: "Request") -> "JSONResponse":
            return _json_response(self.viz_mermaid(
                chain_id=_qp(request, "chain_id"),
                annotation_id=_qp(request, "annotation_id"),
            ))

        async def _viz_timeline(request: "Request") -> "JSONResponse":
            return _json_response(self.viz_echarts_timeline(_qp(request, "debate_log_id")))

        async def _viz_radar(request: "Request") -> "JSONResponse":
            aid = _qp(request, "annotation_id")
            return _json_response(self.viz_echarts_radar(aid))

        async def _viz_all(request: "Request") -> "JSONResponse":
            return _json_response(self.viz_export_all(_qp(request, "target_id")))

        # 9. 健康检查
        async def _health(request: "Request") -> "JSONResponse":
            return _json_response(self.health())

        async def _health_ready(request: "Request") -> "JSONResponse":
            return _json_response(self.health_ready())

        # --- 构建路由列表 ---
        routes = [
            # KPA 标注
            Route("/cc3/annotation/create", _annotation_create, methods=["POST"]),
            Route("/cc3/annotation/list", _annotation_list, methods=["GET"]),
            Route("/cc3/annotation/statistics", _annotation_stats, methods=["GET"]),
            Route("/cc3/annotation/{annotation_id}", _annotation_get, methods=["GET"]),
            Route("/cc3/annotation/{annotation_id}/dimension", _annotation_dimension, methods=["POST"]),
            Route("/cc3/annotation/{annotation_id}/validation", _annotation_validation, methods=["POST"]),
            Route("/cc3/annotation/{annotation_id}/decision", _annotation_decision, methods=["POST"]),
            Route("/cc3/annotation/{annotation_id}/propagation", _annotation_propagation, methods=["POST"]),
            Route("/cc3/annotation/{annotation_id}/rules", _annotation_rules, methods=["POST"]),
            Route("/cc3/annotation/{annotation_id}/completeness", _annotation_completeness, methods=["GET"]),
            Route("/cc3/annotation/{annotation_id}/verify-hash", _annotation_verify_hash, methods=["GET"]),
            Route("/cc3/annotation/{annotation_id}/verify-signature", _annotation_verify_sig, methods=["GET"]),
            Route("/cc3/annotation/{annotation_id}/prov", _annotation_prov, methods=["GET"]),
            # 辩论日志
            Route("/cc3/debate/create", _debate_create, methods=["POST"]),
            Route("/cc3/debate/list", _debate_list, methods=["GET"]),
            Route("/cc3/debate/statistics", _debate_stats, methods=["GET"]),
            Route("/cc3/debate/{log_id}", _debate_get, methods=["GET"]),
            Route("/cc3/debate/{log_id}/round", _debate_round, methods=["POST"]),
            Route("/cc3/debate/{log_id}/convergence", _debate_convergence, methods=["POST"]),
            Route("/cc3/debate/{log_id}/adjudication", _debate_adjudication, methods=["POST"]),
            Route("/cc3/debate/{log_id}/outcome", _debate_outcome, methods=["POST"]),
            Route("/cc3/debate/{log_id}/resource", _debate_resource, methods=["POST"]),
            Route("/cc3/debate/{log_id}/finalize", _debate_finalize, methods=["POST"]),
            Route("/cc3/debate/{log_id}/verify", _debate_verify, methods=["GET"]),
            Route("/cc3/debate/{log_id}/export", _debate_export, methods=["GET"]),
            # 溯源链
            Route("/cc3/chain/create", _chain_create, methods=["POST"]),
            Route("/cc3/chain/list", _chain_list, methods=["GET"]),
            Route("/cc3/chain/audit-verify", _chain_audit, methods=["POST"]),
            Route("/cc3/chain/trace-cross-layer", _chain_trace, methods=["POST"]),
            Route("/cc3/chain/{chain_id}/append", _chain_append, methods=["POST"]),
            Route("/cc3/chain/{chain_id}/verify", _chain_verify, methods=["GET"]),
            Route("/cc3/chain/{chain_id}/verify-node", _chain_verify_node, methods=["GET"]),
            Route("/cc3/chain/{chain_id}/merkle", _chain_merkle, methods=["POST"]),
            Route("/cc3/chain/{chain_id}/merkle-proof", _chain_merkle_proof, methods=["GET"]),
            Route("/cc3/chain/{chain_id}/compress", _chain_compress, methods=["POST"]),
            Route("/cc3/chain/{chain_id}/snapshot", _chain_snapshot, methods=["GET"]),
            # L0 Ledger
            Route("/cc3/ledger/write-kpa", _ledger_write_kpa, methods=["POST"]),
            Route("/cc3/ledger/write-dl", _ledger_write_dl, methods=["POST"]),
            Route("/cc3/ledger/write-cross-layer", _ledger_write_cl, methods=["POST"]),
            Route("/cc3/ledger/write-human-override", _ledger_write_ho, methods=["POST"]),
            Route("/cc3/ledger/query", _ledger_query, methods=["GET"]),
            Route("/cc3/ledger/query-by-time", _ledger_query_time, methods=["GET"]),
            Route("/cc3/ledger/verify", _ledger_verify, methods=["GET"]),
            Route("/cc3/ledger/statistics", _ledger_stats, methods=["GET"]),
            Route("/cc3/ledger/event/{event_id}", _ledger_get_event, methods=["GET"]),
            # 查询引擎
            Route("/cc3/query/trace/{trace_id}", _query_trace, methods=["GET"]),
            Route("/cc3/query/knowledge/{target_id}", _query_knowledge, methods=["GET"]),
            Route("/cc3/query/agent/{agent_id}", _query_agent, methods=["GET"]),
            Route("/cc3/query/timeline", _query_timeline, methods=["GET"]),
            Route("/cc3/query/graph", _query_graph, methods=["GET"]),
            Route("/cc3/query/overview", _query_overview, methods=["GET"]),
            # CC1/CC2 集成
            Route("/cc3/integration/cc1-review", _integ_cc1, methods=["POST"]),
            Route("/cc3/integration/cc2-approval", _integ_cc2, methods=["POST"]),
            Route("/cc3/integration/check-provenance", _integ_prov, methods=["POST"]),
            Route("/cc3/integration/check-escalation", _integ_esc, methods=["POST"]),
            Route("/cc3/integration/check-debate", _integ_deb, methods=["POST"]),
            # KPI 指标
            Route("/cc3/metrics/dashboard", _metrics_dashboard, methods=["GET"]),
            Route("/cc3/metrics/collect-all", _metrics_all, methods=["GET"]),
            Route("/cc3/metrics/coverage", _metrics_coverage, methods=["GET"]),
            Route("/cc3/metrics/integrity", _metrics_integrity, methods=["GET"]),
            Route("/cc3/metrics/performance", _metrics_perf, methods=["GET"]),
            Route("/cc3/metrics/compliance", _metrics_comp, methods=["GET"]),
            Route("/cc3/metrics/export-dashboard", _metrics_export, methods=["GET"]),
            # 可视化
            Route("/cc3/viz/cytoscape", _viz_cyto, methods=["GET"]),
            Route("/cc3/viz/d3-hierarchy", _viz_d3, methods=["GET"]),
            Route("/cc3/viz/mermaid", _viz_mermaid, methods=["GET"]),
            Route("/cc3/viz/echarts-timeline", _viz_timeline, methods=["GET"]),
            Route("/cc3/viz/echarts-radar", _viz_radar, methods=["GET"]),
            Route("/cc3/viz/export-all", _viz_all, methods=["GET"]),
            # 健康检查
            Route("/cc3/health", _health, methods=["GET"]),
            Route("/cc3/health/ready", _health_ready, methods=["GET"]),
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
