"""G6 治理集成 REST API 路由层.

基于 Starlette 构建，将 L0 治理子系统 (G1-G5, CC1-CC2) 暴露为 RESTful JSON API。
完全遵循 L6 API 的 _ok/_err 响应信封、_RouteHandlers 模式和 L6Router 架构。

设计原则 (融合世界先进方案):
- FastAPI DI 启发: 依赖注入式子系统获取，避免全局状态
- OpenAPI 3.1 启发: 结构化错误响应 (RFC 7807 Problem Details)
- URL 路径版本化: /governance/v1/ 前缀
- 三级健康检查: liveness / readiness / deep (Kubernetes 探针模式)
- 全链路 trace_id 传播: 每次请求生成 trace_id，贯穿审计日志
- 线程安全: 所有处理器使用已有引擎的内部锁

端点概览:
    # 健康检查 (三级)
    GET  /governance/v1/health              — 存活探针 (liveness)
    GET  /governance/v1/health/ready        — 就绪探针 (readiness)
    GET  /governance/v1/health/deep         — 深度检查 (全链路连通性)

    # G1/G2 策略治理
    POST   /governance/v1/policies          — 创建策略
    GET    /governance/v1/policies          — 列出策略
    GET    /governance/v1/policies/{id}     — 查询策略
    DELETE /governance/v1/policies/{id}     — 删除策略
    POST   /governance/v1/policies/evaluate — 评估请求
    POST   /governance/v1/policies/evaluate-batch — 批量评估
    GET    /governance/v1/policies/metrics  — 评估度量
    POST   /governance/v1/policies/conflicts — 检测冲突

    # G3 CC1 防幻觉
    POST   /governance/v1/anti-hallucination/verify    — 验证文本
    GET    /governance/v1/anti-hallucination/config     — 获取配置
    PUT    /governance/v1/anti-hallucination/config     — 更新配置
    GET    /governance/v1/anti-hallucination/verifiers  — 列出验证器
    GET    /governance/v1/anti-hallucination/stats      — 统计信息

    # G4 CC2 人机协作
    POST   /governance/v1/collaboration/profiles             — 注册 Agent 配置
    GET    /governance/v1/collaboration/profiles             — 列出配置
    GET    /governance/v1/collaboration/profiles/{id}        — 查询配置
    PUT    /governance/v1/collaboration/profiles/{id}        — 更新配置
    POST   /governance/v1/collaboration/evaluate-react       — REACT 评估
    POST   /governance/v1/collaboration/switch-mode          — 模式切换
    POST   /governance/v1/collaboration/interventions        — 创建干预
    POST   /governance/v1/collaboration/interventions/{id}/respond — 响应干预
    GET    /governance/v1/collaboration/interventions        — 查询干预
    GET    /governance/v1/collaboration/interventions/{id}   — 查询单个干预
    POST   /governance/v1/collaboration/escalate             — 升级到人工
    POST   /governance/v1/collaboration/negotiations         — 发起协商
    POST   /governance/v1/collaboration/negotiations/{id}/rounds — 添加协商轮次
    POST   /governance/v1/collaboration/negotiations/{id}/finalize — 终结协商
    GET    /governance/v1/collaboration/stats                 — 协作统计

    # G5 审计
    GET    /governance/v1/audit/decisions         — 查询决策日志
    GET    /governance/v1/audit/decisions/{id}    — 查询单条决策
    GET    /governance/v1/audit/traces/{id}       — 按 trace 查询
    GET    /governance/v1/audit/aggregate/action   — 按动作聚合
    GET    /governance/v1/audit/aggregate/outcome  — 按结果聚合
    GET    /governance/v1/audit/latency-stats      — 延迟统计
    POST   /governance/v1/audit/baselines          — 构建基线
    POST   /governance/v1/audit/anomalies          — 异常检测
    GET    /governance/v1/audit/alerts             — 告警列表
    GET    /governance/v1/audit/stats              — 审计统计
    GET    /governance/v1/audit/summary            — 审计摘要

    # G5 度量
    POST   /governance/v1/metrics/define           — 定义指标
    POST   /governance/v1/metrics/record           — 记录指标值
    GET    /governance/v1/metrics/{name}/values    — 查询指标值
    GET    /governance/v1/metrics/{name}/latest    — 最新值
    POST   /governance/v1/metrics/aggregate        — 聚合查询
    POST   /governance/v1/metrics/slos             — 注册 SLO
    GET    /governance/v1/metrics/slos/{name}      — 查询 SLO
    POST   /governance/v1/metrics/slos/{name}/evaluate — 评估 SLO
    GET    /governance/v1/metrics/slos/evaluate-all — 评估所有 SLO
    GET    /governance/v1/metrics/slos/alerts      — Burn Rate 告警
    POST   /governance/v1/metrics/dora/deployments — 记录 DORA 部署
    GET    /governance/v1/metrics/dora             — DORA 指标
    GET    /governance/v1/metrics/stats             — 度量统计

    # G5 合规
    POST   /governance/v1/compliance/report        — 生成合规报告
    POST   /governance/v1/compliance/nist-summary  — NIST AI RMF 摘要

    # 全链路集成
    POST   /governance/v1/chain/evaluate           — 全链路评估
    GET    /governance/v1/chain/status              — 链路状态总览
    GET    /governance/v1/routes                   — 路由发现
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from dy3_polaris.l6.core.exceptions import L6Error

_logger = logging.getLogger("dy3_polaris.l0.governance_router")


# ============================================================
# 统一响应 (复用 L6 模式)
# ============================================================


# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err, ok as _ok


def _l6_error_to_dict(err: L6Error) -> dict[str, Any]:
    """将 L6Error 转为响应字典."""
    return _err(-32000, err.code, err.detail)


def _new_trace_id() -> str:
    """生成全链路 trace_id."""
    return f"g6-{uuid.uuid4().hex[:12]}"


async def _parse_body(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """安全解析请求体，返回 (body, error_response)."""
    try:
        body = await request.json()
        return body, None
    except Exception:
        return None, JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)


# ============================================================
# 治理子系统容器
# ============================================================


class GovernanceSubsystems:
    """治理子系统依赖容器 (FastAPI DI 启发).

    聚合所有 G1-G5 + CC1-CC2 子系统实例，
    路由处理器通过此容器获取依赖，避免全局状态。
    """

    def __init__(
        self,
        policy_store: Any | None = None,
        policy_evaluator: Any | None = None,
        anti_hallucination_pipeline: Any | None = None,
        collaboration_engine: Any | None = None,
        audit_engine: Any | None = None,
        metrics_engine: Any | None = None,
        compliance_reporter: Any | None = None,
        review_pipeline: Any | None = None,
    ) -> None:
        self.policy_store = policy_store
        self.policy_evaluator = policy_evaluator
        self.anti_hallucination_pipeline = anti_hallucination_pipeline
        self.collaboration_engine = collaboration_engine
        self.audit_engine = audit_engine
        self.metrics_engine = metrics_engine
        self.compliance_reporter = compliance_reporter
        self.review_pipeline = review_pipeline

    def health_map(self) -> dict[str, bool]:
        """返回各子系统的就绪状态."""
        return {
            "policy_store": self.policy_store is not None,
            "policy_evaluator": self.policy_evaluator is not None,
            "anti_hallucination": self.anti_hallucination_pipeline is not None,
            "collaboration": self.collaboration_engine is not None,
            "audit": self.audit_engine is not None,
            "metrics": self.metrics_engine is not None,
            "compliance": self.compliance_reporter is not None,
            "review_pipeline": self.review_pipeline is not None,
        }


# ============================================================
# 路由处理器
# ============================================================


class _GovernanceHandlers:
    """治理 API 路由处理器.

    遵循 L6 _RouteHandlers 模式:
    1. 从 subsystems 获取所需子系统
    2. 调用子系统方法
    3. 将异常转为统一错误响应
    4. 返回 JSONResponse
    """

    def __init__(self, subsys: GovernanceSubsystems) -> None:
        self._sub = subsys

    # ---- 健康检查 (三级) ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /governance/v1/health — 存活探针."""
        return JSONResponse(_ok({"status": "alive", "timestamp": time.time()}))

    async def health_ready(self, request: Request) -> JSONResponse:
        """GET /governance/v1/health/ready — 就绪探针."""
        hm = self._sub.health_map()
        all_ready = all(hm.values())
        return JSONResponse(_ok({
            "status": "ready" if all_ready else "degraded",
            "subsystems": hm,
        }))

    async def health_deep(self, request: Request) -> JSONResponse:
        """GET /governance/v1/health/deep — 深度检查 (全链路连通性).

        对每个已初始化的子系统执行轻量级操作，验证端到端连通性。
        """
        checks: dict[str, Any] = {}

        # G1 PolicyStore
        if self._sub.policy_store is not None:
            try:
                count = self._sub.policy_store.count()
                checks["policy_store"] = {"status": "ok", "policy_count": count}
            except Exception as e:
                checks["policy_store"] = {"status": "error", "detail": str(e)}

        # G2 PolicyEvaluator
        if self._sub.policy_evaluator is not None:
            try:
                m = self._sub.policy_evaluator.export_metrics()
                checks["policy_evaluator"] = {"status": "ok", "total_evals": m.get("total_evaluations", 0)}
            except Exception as e:
                checks["policy_evaluator"] = {"status": "error", "detail": str(e)}

        # G3 CC1
        if self._sub.anti_hallucination_pipeline is not None:
            try:
                cfg = self._sub.anti_hallucination_pipeline.config
                checks["anti_hallucination"] = {"status": "ok", "verifiers": len(cfg.verifiers)}
            except Exception as e:
                checks["anti_hallucination"] = {"status": "error", "detail": str(e)}

        # G4 CC2
        if self._sub.collaboration_engine is not None:
            try:
                stats = self._sub.collaboration_engine.get_stats()
                checks["collaboration"] = {"status": "ok", "profile_count": stats.get("profile_count", 0)}
            except Exception as e:
                checks["collaboration"] = {"status": "error", "detail": str(e)}

        # G5 Audit
        if self._sub.audit_engine is not None:
            try:
                stats = self._sub.audit_engine.get_stats()
                checks["audit"] = {"status": "ok", "total_decisions": stats.get("total_decisions", 0)}
            except Exception as e:
                checks["audit"] = {"status": "error", "detail": str(e)}

        # G5 Metrics
        if self._sub.metrics_engine is not None:
            try:
                stats = self._sub.metrics_engine.get_stats()
                checks["metrics"] = {"status": "ok", "metric_count": stats.get("metric_count", 0)}
            except Exception as e:
                checks["metrics"] = {"status": "error", "detail": str(e)}

        all_ok = all(c.get("status") == "ok" for c in checks.values())
        return JSONResponse(_ok({
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
        }))

    # ---- G1/G2 策略治理 ----

    async def create_policy(self, request: Request) -> JSONResponse:
        """POST /governance/v1/policies — 创建策略."""
        body, err = await _parse_body(request)
        if err:
            return err
        store = self._sub.policy_store
        if store is None:
            return JSONResponse(_err(-32000, "策略存储未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.models import GovernancePolicy
            policy = GovernancePolicy(**body)
            pid = store.add(policy)
            return JSONResponse(_ok({"policy_id": pid}), status_code=201)
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def list_policies(self, request: Request) -> JSONResponse:
        """GET /governance/v1/policies — 列出策略."""
        store = self._sub.policy_store
        if store is None:
            return JSONResponse(_err(-32000, "策略存储未初始化"), status_code=503)
        try:
            policies = store.list_all()
            return JSONResponse(_ok([p.model_dump(mode="json") for p in policies]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def get_policy(self, request: Request) -> JSONResponse:
        """GET /governance/v1/policies/{id} — 查询策略."""
        store = self._sub.policy_store
        if store is None:
            return JSONResponse(_err(-32000, "策略存储未初始化"), status_code=503)
        pid = request.path_params["id"]
        try:
            policy = store.get(pid)
            if policy is None:
                return JSONResponse(_err(-32000, f"策略未找到: {pid}"), status_code=404)
            return JSONResponse(_ok(policy.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def delete_policy(self, request: Request) -> JSONResponse:
        """DELETE /governance/v1/policies/{id} — 删除策略."""
        store = self._sub.policy_store
        if store is None:
            return JSONResponse(_err(-32000, "策略存储未初始化"), status_code=503)
        pid = request.path_params["id"]
        try:
            ok = store.remove(pid)
            if not ok:
                return JSONResponse(_err(-32000, f"策略未找到: {pid}"), status_code=404)
            return JSONResponse(_ok({"deleted": pid}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def evaluate_policy(self, request: Request) -> JSONResponse:
        """POST /governance/v1/policies/evaluate — 评估请求."""
        body, err = await _parse_body(request)
        if err:
            return err
        evaluator = self._sub.policy_evaluator
        if evaluator is None:
            return JSONResponse(_err(-32000, "策略评估器未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.models import EvalRequest
            req = EvalRequest(**body)
            result = evaluator.evaluate(req)
            return JSONResponse(_ok(result.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def evaluate_batch(self, request: Request) -> JSONResponse:
        """POST /governance/v1/policies/evaluate-batch — 批量评估."""
        body, err = await _parse_body(request)
        if err:
            return err
        evaluator = self._sub.policy_evaluator
        if evaluator is None:
            return JSONResponse(_err(-32000, "策略评估器未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.models import EvalRequest
            requests = [EvalRequest(**r) for r in body.get("requests", [])]
            results = evaluator.evaluate_batch(requests)
            return JSONResponse(_ok([r.model_dump(mode="json") for r in results]))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def policy_metrics(self, request: Request) -> JSONResponse:
        """GET /governance/v1/policies/metrics — 评估度量."""
        evaluator = self._sub.policy_evaluator
        if evaluator is None:
            return JSONResponse(_err(-32000, "策略评估器未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(evaluator.export_metrics()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def detect_conflicts(self, request: Request) -> JSONResponse:
        """POST /governance/v1/policies/conflicts — 检测策略冲突."""
        evaluator = self._sub.policy_evaluator
        if evaluator is None:
            return JSONResponse(_err(-32000, "策略评估器未初始化"), status_code=503)
        try:
            conflicts = evaluator.detect_conflicts()
            return JSONResponse(_ok(conflicts))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    # ---- G3 CC1 防幻觉 ----

    async def cc1_verify(self, request: Request) -> JSONResponse:
        """POST /governance/v1/anti-hallucination/verify — 验证文本."""
        body, err = await _parse_body(request)
        if err:
            return err
        pipeline = self._sub.anti_hallucination_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "防幻觉管道未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc1.models import VerificationRequest
            req = VerificationRequest(**body)
            report = pipeline.verify(req)
            return JSONResponse(_ok(report.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc1_get_config(self, request: Request) -> JSONResponse:
        """GET /governance/v1/anti-hallucination/config — 获取配置."""
        pipeline = self._sub.anti_hallucination_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "防幻觉管道未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(pipeline.config.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc1_update_config(self, request: Request) -> JSONResponse:
        """PUT /governance/v1/anti-hallucination/config — 更新配置."""
        body, err = await _parse_body(request)
        if err:
            return err
        pipeline = self._sub.anti_hallucination_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "防幻觉管道未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc1.models import PipelineConfig
            cfg = PipelineConfig(**body)
            pipeline.update_config(cfg)
            return JSONResponse(_ok({"updated": True}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc1_list_verifiers(self, request: Request) -> JSONResponse:
        """GET /governance/v1/anti-hallucination/verifiers — 列出验证器."""
        pipeline = self._sub.anti_hallucination_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "防幻觉管道未初始化"), status_code=503)
        try:
            reg = pipeline.registry
            verifiers = []
            for name, v in reg._verifiers.items():
                verifiers.append({"name": name, "type": type(v).__name__})
            return JSONResponse(_ok(verifiers))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc1_stats(self, request: Request) -> JSONResponse:
        """GET /governance/v1/anti-hallucination/stats — 统计信息."""
        pipeline = self._sub.anti_hallucination_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "防幻觉管道未初始化"), status_code=503)
        try:
            return JSONResponse(_ok({"pipeline_initialized": True}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    # ---- CC1 四层反幻觉评审引擎 (增强) ----

    async def cc1_review(self, request: Request) -> JSONResponse:
        """POST /governance/v1/review/execute — 执行四层反幻觉评审."""
        body, err = await _parse_body(request)
        if err:
            return err
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc1.models import VerificationRequest
            from dy3_polaris.l0.cc1.review_pipeline import ReviewResult as RR
            req = VerificationRequest(**body)
            result = pipeline.review(req)
            return JSONResponse(_ok(self._review_result_to_dict(result)))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc1_review_layers(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/layers — 列出四层评审规则."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        try:
            layers_info = []
            for layer in [
                pipeline.fact_layer,
                pipeline.logic_layer,
                pipeline.numerical_layer,
                pipeline.provenance_layer,
            ]:
                rules_info = []
                for rule in layer.rules:
                    rules_info.append({
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "description": rule.description,
                        "severity": rule.severity.value,
                    })
                layers_info.append({
                    "layer_type": layer.layer_type.value,
                    "rule_count": len(layer.rules),
                    "rules": rules_info,
                })
            return JSONResponse(_ok(layers_info))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc1_review_config(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/config — 获取评审配置."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        try:
            cfg = pipeline.config
            return JSONResponse(_ok({
                "pass_threshold": cfg.pass_threshold,
                "flag_threshold": cfg.flag_threshold,
                "max_corrections": cfg.max_corrections,
                "enable_self_correction": cfg.enable_self_correction,
                "error_penalty": cfg.error_penalty,
                "critical_penalty": cfg.critical_penalty,
                "warning_penalty": cfg.warning_penalty,
                "info_penalty": cfg.info_penalty,
            }))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc1_review_weights(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/weights — 获取四层权重."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc1.layers import LAYER_WEIGHTS
            weights = {k.value: v for k, v in LAYER_WEIGHTS.items()}
            return JSONResponse(_ok(weights))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc1_review_report(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/reports/{report_id} — 获取评审报告."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        report_id = request.path_params.get("report_id", "")
        result = pipeline.get_result(report_id)
        if result is None:
            return JSONResponse(_err(-32001, f"报告 {report_id} 不存在"), status_code=404)
        return JSONResponse(_ok(self._review_result_to_dict(result)))

    async def cc1_review_reports(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/reports — 列出评审报告."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        agent_id = request.query_params.get("agent_id")
        verdict = request.query_params.get("verdict")
        limit = int(request.query_params.get("limit", "50"))
        from dy3_polaris.l0.cc1.state_machine import ReviewVerdict
        v = ReviewVerdict(verdict) if verdict else None
        results = pipeline.list_results(agent_id=agent_id, verdict=v, limit=limit)
        return JSONResponse(_ok([
            {
                "report_id": r.report_id,
                "agent_id": r.agent_id,
                "verdict": r.verdict.value,
                "composite_score": r.composite_score,
                "created_at": r.created_at,
            }
            for r in results
        ]))

    async def cc1_review_statistics(self, request: Request) -> JSONResponse:
        """GET /governance/v1/review/statistics — 获取评审统计."""
        pipeline = self._sub.review_pipeline
        if pipeline is None:
            return JSONResponse(_err(-32000, "评审管道未初始化"), status_code=503)
        stats = pipeline.get_statistics()
        return JSONResponse(_ok(stats))

    @staticmethod
    def _review_result_to_dict(result: Any) -> dict:
        """将 ReviewResult 转为可序列化字典."""
        data: dict[str, Any] = {
            "report_id": result.report_id,
            "request_id": result.request_id,
            "agent_id": result.agent_id,
            "verdict": result.verdict.value,
            "composite_score": result.composite_score,
            "issues": result.issues,
            "corrected_output": result.corrected_output,
            "created_at": result.created_at,
            "completed_at": result.completed_at,
        }
        # 层结果
        layer_data: dict[str, Any] = {}
        for layer_type, layer_result in result.layer_results.items():
            layer_data[layer_type.value] = {
                "score": layer_result.score,
                "verdict": layer_result.verdict,
                "summary": layer_result.summary,
                "passed_count": layer_result.passed_count,
                "failed_count": layer_result.failed_count,
                "total_count": layer_result.total_count,
                "rule_results": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "passed": r.passed,
                        "severity": r.severity.value,
                        "detail": r.detail,
                        "score": r.score,
                    }
                    for r in layer_result.rule_results
                ],
            }
        data["layer_results"] = layer_data
        data["layer_scores"] = {
            k.value: v for k, v in result.layer_scores.items()
        }
        # 自纠信息
        if result.self_correction:
            sc = result.self_correction
            data["self_correction"] = {
                "attempts": sc.attempts,
                "max_attempts": sc.max_attempts,
                "can_retry": sc.can_retry,
                "needs_escalation": sc.needs_escalation,
                "is_resolved": sc.is_resolved,
                "history": sc.history,
            }
        else:
            data["self_correction"] = None
        return data

    # ---- G4 CC2 人机协作 ----

    async def cc2_register_profile(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/profiles — 注册 Agent 协作配置."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc2.models import AgentCollaborationProfile
            profile = AgentCollaborationProfile(**body)
            engine.register_profile(profile)
            return JSONResponse(_ok({"agent_id": profile.agent_id}), status_code=201)
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_list_profiles(self, request: Request) -> JSONResponse:
        """GET /governance/v1/collaboration/profiles — 列出配置."""
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            profiles = engine.list_profiles()
            return JSONResponse(_ok([p.model_dump(mode="json") for p in profiles]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc2_get_profile(self, request: Request) -> JSONResponse:
        """GET /governance/v1/collaboration/profiles/{id} — 查询配置."""
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        aid = request.path_params["id"]
        try:
            profile = engine.get_profile(aid)
            if profile is None:
                return JSONResponse(_err(-32000, f"Agent 配置未找到: {aid}"), status_code=404)
            return JSONResponse(_ok(profile.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=404)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc2_update_profile(self, request: Request) -> JSONResponse:
        """PUT /governance/v1/collaboration/profiles/{id} — 更新配置."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        aid = request.path_params["id"]
        try:
            updates = {k: v for k, v in body.items() if k != "agent_id"}
            # 枚举字段转换
            if "mode" in updates and isinstance(updates["mode"], str):
                from dy3_polaris.l0.cc2.models import CollaborationMode
                updates["mode"] = CollaborationMode(updates["mode"])
            profile = engine.update_profile(aid, **updates)
            return JSONResponse(_ok(profile.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_evaluate_react(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/evaluate-react — REACT 评估."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc2.models import REACTScore
            agent_id = body["agent_id"]
            score = REACTScore(**body["score"])
            mode = engine.evaluate_react(agent_id, score)
            return JSONResponse(_ok({"agent_id": agent_id, "recommended_mode": mode.value}))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_switch_mode(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/switch-mode — 模式切换."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.cc2.models import CollaborationMode, SwitchTrigger
            # 兼容字符串和枚举
            target = body["target_mode"]
            if isinstance(target, str):
                target = CollaborationMode(target)
            trigger_val = body.get("trigger")
            if trigger_val is not None and isinstance(trigger_val, str):
                trigger_val = SwitchTrigger(trigger_val)
            event = engine.switch_mode(
                agent_id=body["agent_id"],
                to_mode=target,
                reason=body.get("reason", ""),
                trigger=trigger_val,
            )
            return JSONResponse(_ok(event.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_create_intervention(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/interventions — 创建干预."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            record = engine.create_intervention(
                agent_id=body["agent_id"],
                intervention_type=body["intervention_type"],
                reason=body.get("reason", ""),
                context=body.get("context"),
            )
            return JSONResponse(_ok(record.model_dump(mode="json")), status_code=201)
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_respond_intervention(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/interventions/{id}/respond — 响应干预."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        rid = request.path_params["id"]
        try:
            record = engine.respond_to_intervention(
                request_id=rid,
                human_id=body.get("human_id", "human-operator"),
                decision=body["decision"],
                feedback=body.get("human_input", body.get("feedback", "")),
            )
            return JSONResponse(_ok(record.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_list_interventions(self, request: Request) -> JSONResponse:
        """GET /governance/v1/collaboration/interventions — 查询干预."""
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            agent_id = request.query_params.get("agent_id")
            status = request.query_params.get("status")
            results = engine.query_interventions(agent_id=agent_id, status=status)
            return JSONResponse(_ok([r.model_dump(mode="json") for r in results]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc2_get_intervention(self, request: Request) -> JSONResponse:
        """GET /governance/v1/collaboration/interventions/{id} — 查询单个干预."""
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        rid = request.path_params["id"]
        try:
            record = engine.get_intervention(rid)
            if record is None:
                return JSONResponse(_err(-32000, f"干预记录未找到: {rid}"), status_code=404)
            return JSONResponse(_ok(record.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def cc2_escalate(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/escalate — 升级到人工."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            record = engine.escalate_to_human(
                agent_id=body["agent_id"],
                reason=body.get("reason", ""),
                payload=body.get("context"),
            )
            return JSONResponse(_ok(record.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_start_negotiation(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/negotiations — 发起协商."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            session = engine.start_negotiation(
                agent_id=body["agent_id"],
                human_id=body.get("human_id", "human-operator"),
                topic=body.get("proposal", body.get("topic", "")),
                initial_proposal=body.get("context", body.get("initial_proposal", {})),
                max_rounds=body.get("max_rounds"),
            )
            return JSONResponse(_ok(session.model_dump(mode="json")), status_code=201)
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_add_negotiation_round(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/negotiations/{id}/rounds — 添加协商轮次."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        sid = request.path_params["id"]
        try:
            session = engine.add_negotiation_round(
                session_id=sid,
                proposer=body.get("actor", body.get("proposer", "human")),
                proposal=body.get("proposal", {}),
            )
            return JSONResponse(_ok(session.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_finalize_negotiation(self, request: Request) -> JSONResponse:
        """POST /governance/v1/collaboration/negotiations/{id}/finalize — 终结协商."""
        body, err = await _parse_body(request)
        if err:
            return err
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        sid = request.path_params["id"]
        try:
            session = engine.finalize_negotiation(
                session_id=sid,
                decision=body.get("outcome", "approve"),
            )
            return JSONResponse(_ok(session.model_dump(mode="json")))
        except L6Error as e:
            return JSONResponse(_l6_error_to_dict(e), status_code=400)
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def cc2_stats(self, request: Request) -> JSONResponse:
        """GET /governance/v1/collaboration/stats — 协作统计."""
        engine = self._sub.collaboration_engine
        if engine is None:
            return JSONResponse(_err(-32000, "人机协作引擎未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(engine.get_stats()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    # ---- G5 审计 ----

    async def audit_query_decisions(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/decisions — 查询决策日志."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            params = request.query_params
            results = audit.query(
                actor=params.get("actor"),
                action=params.get("action"),
                layer=params.get("layer"),
                outcome=params.get("outcome"),
                agent_id=params.get("agent_id"),
                session_id=params.get("session_id"),
                trace_id=params.get("trace_id"),
                limit=int(params.get("limit", 100)),
            )
            return JSONResponse(_ok([r.model_dump(mode="json") for r in results]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def audit_get_decision(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/decisions/{id} — 查询单条决策."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        did = request.path_params["id"]
        try:
            log = audit.get(did)
            if log is None:
                return JSONResponse(_err(-32000, f"决策日志未找到: {did}"), status_code=404)
            return JSONResponse(_ok(log.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_get_trace(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/traces/{id} — 按 trace 查询."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        tid = request.path_params["id"]
        try:
            logs = audit.get_trace(tid)
            return JSONResponse(_ok([r.model_dump(mode="json") for r in logs]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_aggregate_action(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/aggregate/action — 按动作聚合."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            agent_id = request.query_params.get("agent_id")
            result = audit.aggregate_by_action(agent_id=agent_id or None)
            return JSONResponse(_ok(result))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_aggregate_outcome(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/aggregate/outcome — 按结果聚合."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            agent_id = request.query_params.get("agent_id")
            result = audit.aggregate_by_outcome(agent_id=agent_id or None)
            return JSONResponse(_ok(result))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_latency_stats(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/latency-stats — 延迟统计."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            agent_id = request.query_params.get("agent_id")
            result = audit.latency_stats(agent_id=agent_id or None)
            return JSONResponse(_ok(result))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_build_baseline(self, request: Request) -> JSONResponse:
        """POST /governance/v1/audit/baselines — 构建基线."""
        body, err = await _parse_body(request)
        if err:
            return err
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            baseline = audit.build_baseline(
                entity_id=body.get("entity_id", "global"),
                window_seconds=body.get("window", 3600),
            )
            if baseline is None:
                return JSONResponse(_ok({"baseline": None, "message": "无足够数据构建基线"}))
            return JSONResponse(_ok(baseline.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def audit_detect_anomalies(self, request: Request) -> JSONResponse:
        """POST /governance/v1/audit/anomalies — 异常检测."""
        body, err = await _parse_body(request)
        if err:
            return err
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            alerts = audit.detect_anomalies(
                entity_id=body.get("entity_id", "global"),
                recent_window_seconds=body.get("sensitivity", 300),
            )
            return JSONResponse(_ok([a.model_dump(mode="json") for a in alerts]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def audit_get_alerts(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/alerts — 告警列表."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            alerts = audit.get_alerts()
            return JSONResponse(_ok([a.model_dump(mode="json") for a in alerts]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_stats(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/stats — 审计统计."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(audit.get_stats()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def audit_summary(self, request: Request) -> JSONResponse:
        """GET /governance/v1/audit/summary — 审计摘要."""
        audit = self._sub.audit_engine
        if audit is None:
            return JSONResponse(_err(-32000, "审计引擎未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(audit.export_summary()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    # ---- G5 度量 ----

    async def metrics_define(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/define — 定义指标."""
        body, err = await _parse_body(request)
        if err:
            return err
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.metrics_engine import MetricDefinition, MetricType
            defn = MetricDefinition(
                name=body["name"],
                metric_type=MetricType(body["metric_type"]),
                unit=body.get("unit", ""),
                description=body.get("description", ""),
            )
            metrics.define_metric(defn)
            return JSONResponse(_ok({"defined": body["name"]}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def metrics_record(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/record — 记录指标值."""
        body, err = await _parse_body(request)
        if err:
            return err
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            metrics.record(
                metric_name=body["metric_name"],
                value=float(body["value"]),
                labels=body.get("tags", {}),
            )
            return JSONResponse(_ok({"recorded": body["metric_name"]}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def metrics_get_values(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/{name}/values — 查询指标值."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        name = request.path_params["name"]
        try:
            limit = int(request.query_params.get("limit", 100))
            values = metrics.get_values(name, limit=limit)
            return JSONResponse(_ok([v.model_dump(mode="json") for v in values]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_get_latest(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/{name}/latest — 最新值."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        name = request.path_params["name"]
        try:
            val = metrics.get_latest(name)
            if val is None:
                return JSONResponse(_err(-32000, f"指标无数据: {name}"), status_code=404)
            return JSONResponse(_ok(val.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_aggregate(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/aggregate — 聚合查询."""
        body, err = await _parse_body(request)
        if err:
            return err
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            result = metrics.aggregate(
                metric_name=body["metric_name"],
                func=body["func"],
            )
            return JSONResponse(_ok({"value": result}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def metrics_register_slo(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/slos — 注册 SLO."""
        body, err = await _parse_body(request)
        if err:
            return err
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.metrics_engine import SLODefinition
            slo = SLODefinition(
                name=body["name"],
                target=body["target"],
                metric_name=body["metric_name"],
                window=body.get("window"),
                error_budget_threshold=body.get("error_budget_threshold"),
            )
            metrics.register_slo(slo)
            return JSONResponse(_ok({"registered": body["name"]}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def metrics_get_slo(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/slos/{name} — 查询 SLO."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        name = request.path_params["name"]
        try:
            slo = metrics.get_slo(name)
            if slo is None:
                return JSONResponse(_err(-32000, f"SLO 未找到: {name}"), status_code=404)
            return JSONResponse(_ok(slo.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_evaluate_slo(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/slos/{name}/evaluate — 评估 SLO."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        name = request.path_params["name"]
        try:
            snapshot = metrics.evaluate_slo(name)
            return JSONResponse(_ok(snapshot.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_evaluate_all_slos(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/slos/evaluate-all — 评估所有 SLO."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            snapshots = metrics.evaluate_all_slos()
            return JSONResponse(_ok([s.model_dump(mode="json") for s in snapshots]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_slo_alerts(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/slos/alerts — Burn Rate 告警."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            alerts = metrics.get_burn_rate_alerts()
            return JSONResponse(_ok([a.model_dump(mode="json") for a in alerts]))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_dora_deployment(self, request: Request) -> JSONResponse:
        """POST /governance/v1/metrics/dora/deployments — 记录 DORA 部署."""
        body, err = await _parse_body(request)
        if err:
            return err
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            agent_id = body.get("agent_id", "unknown")
            status = body.get("status", "success")
            duration = body.get("duration_seconds")
            metrics.record_dora_deployment(
                agent_id=agent_id,
                success=(status == "success"),
                latency_ms=(duration * 1000.0 if duration else 0.0),
            )
            return JSONResponse(_ok({"recorded": True}))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def metrics_dora(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/dora — DORA 指标."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            agent_id = request.query_params.get("agent_id")
            result = metrics.get_dora_metrics(agent_id=agent_id or None)
            return JSONResponse(_ok(result))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    async def metrics_stats(self, request: Request) -> JSONResponse:
        """GET /governance/v1/metrics/stats — 度量统计."""
        metrics = self._sub.metrics_engine
        if metrics is None:
            return JSONResponse(_err(-32000, "度量引擎未初始化"), status_code=503)
        try:
            return JSONResponse(_ok(metrics.get_stats()))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=500)

    # ---- G5 合规 ----

    async def compliance_report(self, request: Request) -> JSONResponse:
        """POST /governance/v1/compliance/report — 生成合规报告."""
        body, err = await _parse_body(request)
        if err:
            return err
        reporter = self._sub.compliance_reporter
        if reporter is None:
            return JSONResponse(_err(-32000, "合规报告器未初始化"), status_code=503)
        try:
            report = reporter.generate_from_audit(
                audit_stats=body.get("audit_stats", {}),
                metrics_stats=body.get("metrics_stats", {}),
                frameworks=body.get("frameworks", ["SOC2", "NIST_AI_RMF"]),
            )
            return JSONResponse(_ok(report.model_dump(mode="json")))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    async def compliance_nist_summary(self, request: Request) -> JSONResponse:
        """POST /governance/v1/compliance/nist-summary — NIST AI RMF 摘要."""
        body, err = await _parse_body(request)
        if err:
            return err
        reporter = self._sub.compliance_reporter
        if reporter is None:
            return JSONResponse(_err(-32000, "合规报告器未初始化"), status_code=503)
        try:
            from dy3_polaris.l0.governance.compliance import GovernanceComplianceReport
            report = GovernanceComplianceReport(**body["report"])
            summary = reporter.generate_nist_summary(report)
            return JSONResponse(_ok(summary))
        except Exception as e:
            return JSONResponse(_err(-32000, str(e)), status_code=400)

    # ---- 全链路集成 ----

    async def chain_evaluate(self, request: Request) -> JSONResponse:
        """POST /governance/v1/chain/evaluate — 全链路评估.

        模拟一次完整的治理链路:
        1. G1/G2: 策略评估
        2. G3: CC1 防幻觉验证
        3. G4: CC2 REACT 评估 (如果策略要求)
        4. G5: 审计日志记录
        5. G5: 度量指标记录
        6. G5: 合规报告生成

        所有操作在同一个 trace_id 下执行。
        """
        body, err = await _parse_body(request)
        if err:
            return err

        trace_id = body.get("trace_id") or _new_trace_id()
        results: dict[str, Any] = {"trace_id": trace_id, "stages": {}}

        # 阶段 1: G1/G2 策略评估
        if self._sub.policy_evaluator is not None and "policy" in body:
            try:
                from dy3_polaris.l0.governance.models import EvalRequest
                req = EvalRequest(**body["policy"])
                eval_result = self._sub.policy_evaluator.evaluate(req)
                results["stages"]["policy_eval"] = eval_result.model_dump(mode="json")
            except Exception as e:
                results["stages"]["policy_eval"] = {"error": str(e)}

        # 阶段 2: G3 CC1 防幻觉
        if self._sub.anti_hallucination_pipeline is not None and "anti_hallucination" in body:
            try:
                from dy3_polaris.l0.cc1.models import VerificationRequest
                req = VerificationRequest(**body["anti_hallucination"])
                report = self._sub.anti_hallucination_pipeline.verify(req)
                results["stages"]["anti_hallucination"] = report.model_dump(mode="json")
            except Exception as e:
                results["stages"]["anti_hallucination"] = {"error": str(e)}

        # 阶段 3: G4 CC2 REACT 评估
        if self._sub.collaboration_engine is not None and "collaboration" in body:
            try:
                from dy3_polaris.l0.cc2.models import REACTScore
                cc2_body = body["collaboration"]
                score = REACTScore(**cc2_body["score"])
                mode = self._sub.collaboration_engine.evaluate_react(
                    cc2_body["agent_id"], score,
                )
                results["stages"]["collaboration"] = {
                    "agent_id": cc2_body["agent_id"],
                    "recommended_mode": mode.value,
                }
            except Exception as e:
                results["stages"]["collaboration"] = {"error": str(e)}

        # 阶段 4: G5 审计记录
        if self._sub.audit_engine is not None:
            try:
                self._sub.audit_engine.record(
                    actor=body.get("actor", "chain-api"),
                    action="chain_evaluate",
                    layer="G6",
                    trace_id=trace_id,
                    outcome="success",
                    input_context=body,
                    output_result=results,
                )
                results["stages"]["audit"] = {"recorded": True}
            except Exception as e:
                results["stages"]["audit"] = {"error": str(e)}

        # 阶段 5: G5 度量记录
        if self._sub.metrics_engine is not None:
            try:
                self._sub.metrics_engine.record(
                    metric_name="chain_evaluate_total",
                    value=1.0,
                    labels={"trace_id": trace_id},
                )
                results["stages"]["metrics"] = {"recorded": True}
            except Exception as e:
                results["stages"]["metrics"] = {"error": str(e)}

        return JSONResponse(_ok(results))

    async def chain_status(self, request: Request) -> JSONResponse:
        """GET /governance/v1/chain/status — 链路状态总览."""
        hm = self._sub.health_map()
        stats: dict[str, Any] = {"subsystems": hm}

        if self._sub.audit_engine is not None:
            try:
                stats["audit"] = self._sub.audit_engine.get_stats()
            except Exception:
                pass

        if self._sub.metrics_engine is not None:
            try:
                stats["metrics"] = self._sub.metrics_engine.get_stats()
            except Exception:
                pass

        if self._sub.collaboration_engine is not None:
            try:
                stats["collaboration"] = self._sub.collaboration_engine.get_stats()
            except Exception:
                pass

        return JSONResponse(_ok(stats))

    async def routes_discovery(self, request: Request) -> JSONResponse:
        """GET /governance/v1/routes — 路由发现."""
        router = GovernanceRouter(self._sub)
        return JSONResponse(_ok(router.get_routes_summary()))


# ============================================================
# GovernanceRouter
# ============================================================


class GovernanceRouter:
    """治理 REST API 路由器.

    将 L0 治理子系统 (G1-G5, CC1-CC2) 暴露为 RESTful API。
    基于 Starlette 构建，完全遵循 L6 L6Router 架构。

    Usage::

        from dy3_polaris.l0.governance import PolicyStore, PolicyEvaluator
        from dy3_polaris.l0.cc1.pipeline import AntiHallucinationPipeline
        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline
        from dy3_polaris.l0.cc2.engine import CollaborationEngine
        from dy3_polaris.l0.governance import AuditEngine, MetricsEngine, ComplianceReporter

        subsys = GovernanceSubsystems(
            policy_store=PolicyStore(),
            policy_evaluator=PolicyEvaluator(store),
            anti_hallucination_pipeline=AntiHallucinationPipeline(),
            review_pipeline=ReviewPipeline(),
            collaboration_engine=CollaborationEngine(),
            audit_engine=AuditEngine(),
            metrics_engine=MetricsEngine(),
            compliance_reporter=ComplianceReporter(),
        )
        router = GovernanceRouter(subsys)
        app = router.create_app()

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)

        # 或挂载到 L6 Router
        # from dy3_polaris.l6.api.router import L6Router
        # l6_app = L6Router(engine, config).create_app()
        # l6_app.mount("/governance", app)
    """

    def __init__(self, subsys: GovernanceSubsystems) -> None:
        self._subsys = subsys
        self._handlers = _GovernanceHandlers(subsys)

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例.

        注意: 该子应用由 UnifiedApp 挂载于 ``/governance`` 前缀下,
        故内部路径从 ``/v1`` 开始, 完整路径为 ``/governance/v1/...``。
        """
        h = self._handlers
        p = "/v1"

        routes = [
            # 健康检查
            Route(f"{p}/health", h.health, methods=["GET"]),
            Route(f"{p}/health/ready", h.health_ready, methods=["GET"]),
            Route(f"{p}/health/deep", h.health_deep, methods=["GET"]),

            # G1/G2 策略治理
            Route(f"{p}/policies", h.create_policy, methods=["POST"]),
            Route(f"{p}/policies", h.list_policies, methods=["GET"]),
            Route(f"{p}/policies/conflicts", h.detect_conflicts, methods=["POST"]),
            Route(f"{p}/policies/metrics", h.policy_metrics, methods=["GET"]),
            Route(f"{p}/policies/evaluate", h.evaluate_policy, methods=["POST"]),
            Route(f"{p}/policies/evaluate-batch", h.evaluate_batch, methods=["POST"]),
            Route(f"{p}/policies/{{id}}", h.get_policy, methods=["GET"]),
            Route(f"{p}/policies/{{id}}", h.delete_policy, methods=["DELETE"]),

            # G3 CC1 防幻觉
            Route(f"{p}/anti-hallucination/verify", h.cc1_verify, methods=["POST"]),
            Route(f"{p}/anti-hallucination/config", h.cc1_get_config, methods=["GET"]),
            Route(f"{p}/anti-hallucination/config", h.cc1_update_config, methods=["PUT"]),
            Route(f"{p}/anti-hallucination/verifiers", h.cc1_list_verifiers, methods=["GET"]),
            Route(f"{p}/anti-hallucination/stats", h.cc1_stats, methods=["GET"]),

            # CC1 四层反幻觉评审引擎 (增强)
            Route(f"{p}/review/execute", h.cc1_review, methods=["POST"]),
            Route(f"{p}/review/layers", h.cc1_review_layers, methods=["GET"]),
            Route(f"{p}/review/config", h.cc1_review_config, methods=["GET"]),
            Route(f"{p}/review/weights", h.cc1_review_weights, methods=["GET"]),
            Route(f"{p}/review/reports", h.cc1_review_reports, methods=["GET"]),
            Route(f"{p}/review/reports/{{report_id}}", h.cc1_review_report, methods=["GET"]),
            Route(f"{p}/review/statistics", h.cc1_review_statistics, methods=["GET"]),

            # G4 CC2 人机协作
            Route(f"{p}/collaboration/profiles", h.cc2_register_profile, methods=["POST"]),
            Route(f"{p}/collaboration/profiles", h.cc2_list_profiles, methods=["GET"]),
            Route(f"{p}/collaboration/profiles/{{id}}", h.cc2_get_profile, methods=["GET"]),
            Route(f"{p}/collaboration/profiles/{{id}}", h.cc2_update_profile, methods=["PUT"]),
            Route(f"{p}/collaboration/evaluate-react", h.cc2_evaluate_react, methods=["POST"]),
            Route(f"{p}/collaboration/switch-mode", h.cc2_switch_mode, methods=["POST"]),
            Route(f"{p}/collaboration/interventions", h.cc2_create_intervention, methods=["POST"]),
            Route(f"{p}/collaboration/interventions", h.cc2_list_interventions, methods=["GET"]),
            Route(f"{p}/collaboration/interventions/{{id}}", h.cc2_get_intervention, methods=["GET"]),
            Route(f"{p}/collaboration/interventions/{{id}}/respond", h.cc2_respond_intervention, methods=["POST"]),
            Route(f"{p}/collaboration/escalate", h.cc2_escalate, methods=["POST"]),
            Route(f"{p}/collaboration/negotiations", h.cc2_start_negotiation, methods=["POST"]),
            Route(f"{p}/collaboration/negotiations/{{id}}/rounds", h.cc2_add_negotiation_round, methods=["POST"]),
            Route(f"{p}/collaboration/negotiations/{{id}}/finalize", h.cc2_finalize_negotiation, methods=["POST"]),
            Route(f"{p}/collaboration/stats", h.cc2_stats, methods=["GET"]),

            # G5 审计
            Route(f"{p}/audit/decisions", h.audit_query_decisions, methods=["GET"]),
            Route(f"{p}/audit/decisions/{{id}}", h.audit_get_decision, methods=["GET"]),
            Route(f"{p}/audit/traces/{{id}}", h.audit_get_trace, methods=["GET"]),
            Route(f"{p}/audit/aggregate/action", h.audit_aggregate_action, methods=["GET"]),
            Route(f"{p}/audit/aggregate/outcome", h.audit_aggregate_outcome, methods=["GET"]),
            Route(f"{p}/audit/latency-stats", h.audit_latency_stats, methods=["GET"]),
            Route(f"{p}/audit/baselines", h.audit_build_baseline, methods=["POST"]),
            Route(f"{p}/audit/anomalies", h.audit_detect_anomalies, methods=["POST"]),
            Route(f"{p}/audit/alerts", h.audit_get_alerts, methods=["GET"]),
            Route(f"{p}/audit/stats", h.audit_stats, methods=["GET"]),
            Route(f"{p}/audit/summary", h.audit_summary, methods=["GET"]),

            # G5 度量
            Route(f"{p}/metrics/define", h.metrics_define, methods=["POST"]),
            Route(f"{p}/metrics/record", h.metrics_record, methods=["POST"]),
            Route(f"{p}/metrics/aggregate", h.metrics_aggregate, methods=["POST"]),
            Route(f"{p}/metrics/{{name}}/values", h.metrics_get_values, methods=["GET"]),
            Route(f"{p}/metrics/{{name}}/latest", h.metrics_get_latest, methods=["GET"]),
            Route(f"{p}/metrics/slos", h.metrics_register_slo, methods=["POST"]),
            Route(f"{p}/metrics/slos/evaluate-all", h.metrics_evaluate_all_slos, methods=["GET"]),
            Route(f"{p}/metrics/slos/alerts", h.metrics_slo_alerts, methods=["GET"]),
            Route(f"{p}/metrics/slos/{{name}}", h.metrics_get_slo, methods=["GET"]),
            Route(f"{p}/metrics/slos/{{name}}/evaluate", h.metrics_evaluate_slo, methods=["POST"]),
            Route(f"{p}/metrics/dora/deployments", h.metrics_dora_deployment, methods=["POST"]),
            Route(f"{p}/metrics/dora", h.metrics_dora, methods=["GET"]),
            Route(f"{p}/metrics/stats", h.metrics_stats, methods=["GET"]),

            # G5 合规
            Route(f"{p}/compliance/report", h.compliance_report, methods=["POST"]),
            Route(f"{p}/compliance/nist-summary", h.compliance_nist_summary, methods=["POST"]),

            # 全链路集成
            Route(f"{p}/chain/evaluate", h.chain_evaluate, methods=["POST"]),
            Route(f"{p}/chain/status", h.chain_status, methods=["GET"]),
            Route(f"{p}/routes", h.routes_discovery, methods=["GET"]),
        ]

        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ]

        app = Starlette(routes=routes, middleware=middleware)
        return app

    def get_routes_summary(self) -> list[dict[str, str]]:
        """获取所有路由摘要 (用于文档/发现)."""
        p = "/governance/v1"
        return [
            # 健康检查
            {"path": f"{p}/health", "methods": ["GET"], "description": "存活探针 (liveness)"},
            {"path": f"{p}/health/ready", "methods": ["GET"], "description": "就绪探针 (readiness)"},
            {"path": f"{p}/health/deep", "methods": ["GET"], "description": "深度检查 (全链路连通性)"},
            # G1/G2
            {"path": f"{p}/policies", "methods": ["GET", "POST"], "description": "列出/创建策略"},
            {"path": f"{p}/policies/{{id}}", "methods": ["GET", "DELETE"], "description": "查询/删除策略"},
            {"path": f"{p}/policies/evaluate", "methods": ["POST"], "description": "策略评估"},
            {"path": f"{p}/policies/evaluate-batch", "methods": ["POST"], "description": "批量策略评估"},
            {"path": f"{p}/policies/metrics", "methods": ["GET"], "description": "评估度量"},
            {"path": f"{p}/policies/conflicts", "methods": ["POST"], "description": "检测策略冲突"},
            # G3 CC1
            {"path": f"{p}/anti-hallucination/verify", "methods": ["POST"], "description": "防幻觉验证"},
            {"path": f"{p}/anti-hallucination/config", "methods": ["GET", "PUT"], "description": "获取/更新配置"},
            {"path": f"{p}/anti-hallucination/verifiers", "methods": ["GET"], "description": "列出验证器"},
            {"path": f"{p}/anti-hallucination/stats", "methods": ["GET"], "description": "统计信息"},
            # CC1 四层反幻觉评审引擎 (增强)
            {"path": f"{p}/review/execute", "methods": ["POST"], "description": "执行四层反幻觉评审"},
            {"path": f"{p}/review/layers", "methods": ["GET"], "description": "列出四层评审规则"},
            {"path": f"{p}/review/config", "methods": ["GET"], "description": "获取评审配置"},
            {"path": f"{p}/review/weights", "methods": ["GET"], "description": "获取四层权重"},
            {"path": f"{p}/review/reports", "methods": ["GET"], "description": "列出评审报告"},
            {"path": f"{p}/review/reports/{{report_id}}", "methods": ["GET"], "description": "获取指定评审报告"},
            {"path": f"{p}/review/statistics", "methods": ["GET"], "description": "获取评审统计"},
            # G4 CC2
            {"path": f"{p}/collaboration/profiles", "methods": ["GET", "POST"], "description": "列出/注册协作配置"},
            {"path": f"{p}/collaboration/profiles/{{id}}", "methods": ["GET", "PUT"], "description": "查询/更新配置"},
            {"path": f"{p}/collaboration/evaluate-react", "methods": ["POST"], "description": "REACT 评估"},
            {"path": f"{p}/collaboration/switch-mode", "methods": ["POST"], "description": "模式切换"},
            {"path": f"{p}/collaboration/interventions", "methods": ["GET", "POST"], "description": "查询/创建干预"},
            {"path": f"{p}/collaboration/interventions/{{id}}", "methods": ["GET"], "description": "查询单个干预"},
            {"path": f"{p}/collaboration/interventions/{{id}}/respond", "methods": ["POST"], "description": "响应干预"},
            {"path": f"{p}/collaboration/escalate", "methods": ["POST"], "description": "升级到人工"},
            {"path": f"{p}/collaboration/negotiations", "methods": ["POST"], "description": "发起协商"},
            {"path": f"{p}/collaboration/negotiations/{{id}}/rounds", "methods": ["POST"], "description": "添加协商轮次"},
            {"path": f"{p}/collaboration/negotiations/{{id}}/finalize", "methods": ["POST"], "description": "终结协商"},
            {"path": f"{p}/collaboration/stats", "methods": ["GET"], "description": "协作统计"},
            # G5 审计
            {"path": f"{p}/audit/decisions", "methods": ["GET"], "description": "查询决策日志"},
            {"path": f"{p}/audit/decisions/{{id}}", "methods": ["GET"], "description": "查询单条决策"},
            {"path": f"{p}/audit/traces/{{id}}", "methods": ["GET"], "description": "按 trace 查询"},
            {"path": f"{p}/audit/aggregate/action", "methods": ["GET"], "description": "按动作聚合"},
            {"path": f"{p}/audit/aggregate/outcome", "methods": ["GET"], "description": "按结果聚合"},
            {"path": f"{p}/audit/latency-stats", "methods": ["GET"], "description": "延迟统计"},
            {"path": f"{p}/audit/baselines", "methods": ["POST"], "description": "构建基线"},
            {"path": f"{p}/audit/anomalies", "methods": ["POST"], "description": "异常检测"},
            {"path": f"{p}/audit/alerts", "methods": ["GET"], "description": "告警列表"},
            {"path": f"{p}/audit/stats", "methods": ["GET"], "description": "审计统计"},
            {"path": f"{p}/audit/summary", "methods": ["GET"], "description": "审计摘要"},
            # G5 度量
            {"path": f"{p}/metrics/define", "methods": ["POST"], "description": "定义指标"},
            {"path": f"{p}/metrics/record", "methods": ["POST"], "description": "记录指标值"},
            {"path": f"{p}/metrics/{{name}}/values", "methods": ["GET"], "description": "查询指标值"},
            {"path": f"{p}/metrics/{{name}}/latest", "methods": ["GET"], "description": "最新指标值"},
            {"path": f"{p}/metrics/aggregate", "methods": ["POST"], "description": "聚合查询"},
            {"path": f"{p}/metrics/slos", "methods": ["POST"], "description": "注册 SLO"},
            {"path": f"{p}/metrics/slos/{{name}}", "methods": ["GET"], "description": "查询 SLO"},
            {"path": f"{p}/metrics/slos/{{name}}/evaluate", "methods": ["POST"], "description": "评估 SLO"},
            {"path": f"{p}/metrics/slos/evaluate-all", "methods": ["GET"], "description": "评估所有 SLO"},
            {"path": f"{p}/metrics/slos/alerts", "methods": ["GET"], "description": "Burn Rate 告警"},
            {"path": f"{p}/metrics/dora/deployments", "methods": ["POST"], "description": "记录 DORA 部署"},
            {"path": f"{p}/metrics/dora", "methods": ["GET"], "description": "DORA 指标"},
            {"path": f"{p}/metrics/stats", "methods": ["GET"], "description": "度量统计"},
            # G5 合规
            {"path": f"{p}/compliance/report", "methods": ["POST"], "description": "生成合规报告"},
            {"path": f"{p}/compliance/nist-summary", "methods": ["POST"], "description": "NIST AI RMF 摘要"},
            # 全链路
            {"path": f"{p}/chain/evaluate", "methods": ["POST"], "description": "全链路评估"},
            {"path": f"{p}/chain/status", "methods": ["GET"], "description": "链路状态总览"},
            {"path": f"{p}/routes", "methods": ["GET"], "description": "路由发现"},
        ]


def create_governance_app(
    *,
    include_review_pipeline: bool = True,
) -> Starlette:
    """创建预配置的治理 Starlette 应用.

    便捷工厂函数, 自动初始化所有子系统 (G1-G5, CC1-CC2),
    包括四层反幻觉评审引擎.

    Args:
        include_review_pipeline: 是否初始化四层评审管道 (默认 True)

    Returns:
        Starlette 应用实例

    Usage::

        from dy3_polaris.l0.governance_router import create_governance_app

        app = create_governance_app()
        # 或挂载到 L6 Router
        # l6_app.mount("/governance", app)
    """
    from dy3_polaris.l0.governance import (
        AuditEngine,
        ComplianceReporter,
        MetricsEngine,
        PolicyEvaluator,
        PolicyStore,
    )
    from dy3_polaris.l0.cc1.pipeline import AntiHallucinationPipeline
    from dy3_polaris.l0.cc2.engine import CollaborationEngine

    store = PolicyStore()

    kwargs: dict[str, Any] = dict(
        policy_store=store,
        policy_evaluator=PolicyEvaluator(store),
        anti_hallucination_pipeline=AntiHallucinationPipeline(),
        collaboration_engine=CollaborationEngine(),
        audit_engine=AuditEngine(),
        metrics_engine=MetricsEngine(),
        compliance_reporter=ComplianceReporter(),
    )

    if include_review_pipeline:
        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline

        kwargs["review_pipeline"] = ReviewPipeline()

    subsys = GovernanceSubsystems(**kwargs)
    router = GovernanceRouter(subsys)
    app = router.create_app()
    # 与 UnifiedApp 一致: 挂载到 /governance 前缀, 内部 /v1
    from starlette.routing import Mount

    wrapped = Starlette(routes=[Mount("/governance", app=app)])
    return wrapped


__all__ = [
    "GovernanceSubsystems",
    "GovernanceRouter",
    "_GovernanceHandlers",
    "create_governance_app",
    "_ok",
    "_err",
    "_l6_error_to_dict",
]
