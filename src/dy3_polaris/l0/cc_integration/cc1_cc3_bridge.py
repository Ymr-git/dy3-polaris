"""CC4 三横切集成 — CC1→CC3 桥接器.

将 CC1 (四层反幻觉评审) 的评审结果自动标注到 CC3 (溯源捕获) 的 KPA 校验维度,
打通 "评审 → 溯源" 的单向数据流, 形成不可篡改的评审溯源链路.

桥接流程::

    ReviewResult (CC1)
        │  layer_scores 枚举键 → 字符串键
        │  verdict → CC3 verdict
        ▼
    KPAAnnotation (CC3)  ←─ _ensure_annotation (创建 / 复用)
        │  CCIntegration.on_cc1_review_completed()
        │  → 更新校验维度 + 写入 Ledger + 追加溯源链节点
        ▼
    BridgeEvent (审计事件, CloudEvents 格式)

键值映射:
    ┌──────────────────────────────┬─────────────────────┐
    │ ReviewLayerType              │ CC3 校验维度键       │
    ├──────────────────────────────┼─────────────────────┤
    │ L1_FACT                      │ "factual"           │
    │ L2_LOGIC                     │ "logical"           │
    │ L3_NUMERICAL                 │ "numerical"         │
    │ L4_PROVENANCE                │ "provenance"        │
    └──────────────────────────────┴─────────────────────┘

    ┌──────────────────────────────┬─────────────────────┐
    │ ReviewVerdict (CC1)          │ verdict (CC3)       │
    ├──────────────────────────────┼─────────────────────┤
    │ PASS                         │ "pass"              │
    │ FLAG                         │ "pass_with_notes"   │
    │ BLOCK                        │ "fail"              │
    └──────────────────────────────┴─────────────────────┘

断路器保护:
- CC3 集成调用 (on_cc1_review_completed) 经 CircuitBreaker 保护
- CC3 连续失败达阈值时断路器跳闸, 桥接降级返回错误事件
- 避免级联故障影响 CC1 评审主流程

共享实例约束:
- 桥接器持有的 ``KPAEngine`` 与 ``CCIntegration`` 内部 ``KPAEngine``
  始终为同一实例, 确保 "标注创建" 与 "校验维度更新" 落在同一存储,
  避免 KPA 标注孤立导致更新失败.

融合世界先进方案:
- Service Mesh (Istio): 横切关注点统一编排 + 断路器隔离
- Event-Driven Architecture: CloudEvents 标准化桥接事件
- OpenTelemetry: trace_id 全链路传递
- W3C PROV: 评审 Activity → KPA Entity 关联映射
- Hystrix / Resilience4j: 断路器三态保护
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import BridgeDirection, BridgeEvent
from .exceptions import BridgeConnectionError, CircuitBreakerOpenError
from .circuit_breaker import CircuitBreaker
from ..cc1.review_pipeline import ReviewResult, ReviewVerdict, ReviewLayerType
from ..cc3.cc_integration import CCIntegration
from ..cc3.kpa_engine import KPAEngine
from ..cc3.models import TargetType, ValidationVerdict

logger = logging.getLogger(__name__)


#: ReviewLayerType 枚举键 → CC3 校验维度字符串键.
#:
#: CC1 以 ``ReviewLayerType`` 枚举作为 layer_scores 的键,
#: CC3 校验维度 (ValidationDimension.four_layer_scores) 使用
#: factual / logical / numerical / provenance 字符串键.
LAYER_SCORE_KEY_MAP: dict[ReviewLayerType, str] = {
    ReviewLayerType.L1_FACT: "factual",
    ReviewLayerType.L2_LOGIC: "logical",
    ReviewLayerType.L3_NUMERICAL: "numerical",
    ReviewLayerType.L4_PROVENANCE: "provenance",
}

#: ReviewLayerType 枚举值 → CC3 校验维度字符串键 (兼容字符串键输入).
LAYER_SCORE_VALUE_MAP: dict[str, str] = {
    layer.value: name for layer, name in LAYER_SCORE_KEY_MAP.items()
}

#: CC1 ReviewVerdict → CC3 verdict 字符串.
#:
#: PASS 直接通过; FLAG 转为带备注通过 (pass_with_notes);
#: BLOCK 转为失败 (fail).
VERDICT_MAP: dict[ReviewVerdict, str] = {
    ReviewVerdict.PASS: "pass",
    ReviewVerdict.FLAG: "pass_with_notes",
    ReviewVerdict.BLOCK: "fail",
}

#: CC3 合法 verdict 值集合 (源自 ValidationVerdict 枚举).
VALID_CC3_VERDICTS: frozenset[str] = frozenset(
    v.value for v in ValidationVerdict
)


class CC1CC3Bridge:
    """CC1→CC3 桥接器 — 评审结果自动标注到 KPA 校验维度.

    将 CC1 四层反幻觉评审产出的 :class:`ReviewResult` 自动转换为 CC3 KPA
    标注的校验维度数据, 并通过 :meth:`CCIntegration.on_cc1_review_completed`
    写入 KPA、Ledger 与溯源链.

    核心职责:
        1. 确保 KPA 标注存在 (annotation_id 为 None 时自动创建)
        2. ReviewLayerType 枚举键 → CC3 字符串键 (factual/logical/...)
        3. ReviewVerdict → CC3 verdict (pass/pass_with_notes/fail)
        4. 提取问题列表与自纠回路迭代次数
        5. 断路器保护的 CC3 集成调用
        6. 桥接事件审计 (CloudEvents 格式)

    使用示例::

        from dy3_polaris.l0.cc1.review_pipeline import ReviewPipeline
        from dy3_polaris.l0.cc_integration import CC1CC3Bridge
        from dy3_polaris.l0.cc3.models import TargetType

        pipeline = ReviewPipeline()
        result = pipeline.review(request)

        bridge = CC1CC3Bridge()
        outcome = bridge.bridge(
            review_result=result,
            target_id="kp-dy3-emission",
            target_type=TargetType.KNOWLEDGE_POINT,
            trace_id="trace-001",
        )

        if outcome["success"]:
            print("KPA 标注:", outcome["annotation_id"])
            print("完整度:", outcome["completeness"])

    Note:
        桥接器持有的 ``KPAEngine`` 与 ``CCIntegration`` 内部 ``KPAEngine``
        始终为同一实例 (在 ``__init__`` 中保证). 本桥接器在 ``bridge``
        调用期间非线程安全, 并发场景下请为每个线程 / 任务创建独立实例.
    """

    #: 事件日志上限 (超出后保留最近一半)
    _MAX_EVENTS: int = 1000

    def __init__(
        self,
        cc_integration: CCIntegration | None = None,
        kpa_engine: KPAEngine | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """初始化 CC1→CC3 桥接器.

        所有依赖项为 None 时自动创建默认实例, 开箱即用.
        ``KPAEngine`` 与 ``CCIntegration`` 共享同一实例, 避免标注孤立.

        Args:
            cc_integration: CC3 跨切面集成器, 为 None 时自动创建.
            kpa_engine: KPA 标注引擎, 为 None 时从 ``cc_integration``
                复用或创建. 必须与 ``cc_integration`` 内部的 KPAEngine
                为同一实例, 否则标注创建与更新会落在不同存储.
            circuit_breaker: 断路器, 为 None 时创建保护 CC3 的默认实例.
        """
        # 确保 KPAEngine 与 CCIntegration 共享同一实例
        if kpa_engine is None and cc_integration is not None:
            kpa_engine = getattr(cc_integration, "_kpa", None)
        if kpa_engine is None:
            kpa_engine = KPAEngine()
        if cc_integration is None:
            cc_integration = CCIntegration(kpa_engine=kpa_engine)

        # 防御性检查: 显式传入两者时须为同一实例
        if cc_integration is not None:
            internal_kpa = getattr(cc_integration, "_kpa", None)
            if internal_kpa is not None and internal_kpa is not kpa_engine:
                logger.warning(
                    "kpa_engine 与 cc_integration 内部 KPAEngine 不是同一实例, "
                    "可能导致 KPA 标注查找失败"
                )

        self._cc_integration: CCIntegration = cc_integration
        self._kpa_engine: KPAEngine = kpa_engine
        self._circuit_breaker: CircuitBreaker = (
            circuit_breaker or CircuitBreaker(module="cc3")
        )

        # 桥接事件审计日志
        self._events: list[BridgeEvent] = []

        # 桥接统计
        self._stats: dict[str, Any] = {
            "total_bridges": 0,
            "successful_bridges": 0,
            "failed_bridges": 0,
            "circuit_breaker_trips": 0,
            "annotations_created": 0,
            "by_verdict": {
                "pass": 0,
                "pass_with_notes": 0,
                "fail": 0,
            },
            "total_latency_ms_sum": 0.0,
        }

    # ========================================================
    # 属性
    # ========================================================

    @property
    def cc_integration(self) -> CCIntegration:
        """CC3 跨切面集成器."""
        return self._cc_integration

    @property
    def kpa_engine(self) -> KPAEngine:
        """KPA 标注引擎."""
        return self._kpa_engine

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """断路器."""
        return self._circuit_breaker

    # ========================================================
    # 核心桥接方法
    # ========================================================

    def bridge(
        self,
        review_result: ReviewResult,
        target_id: str = "",
        target_type: TargetType = TargetType.CONTENT,
        annotation_id: str | None = None,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """将 CC1 评审结果桥接到 CC3 KPA 校验维度.

        执行流程:
            1. 确保 KPA 标注存在 (annotation_id 为 None 时自动创建)
            2. 转换 layer_scores (ReviewLayerType 枚举键 → 字符串键)
            3. 转换 verdict (ReviewVerdict → CC3 verdict 字符串)
            4. 提取问题列表与自纠回路迭代次数
            5. 经断路器调用 :meth:`CCIntegration.on_cc1_review_completed`
            6. 记录 :class:`BridgeEvent` 审计事件并返回结果

        Args:
            review_result: CC1 评审结果.
            target_id: 标注对象 ID (创建新标注时使用).
            target_type: 标注对象类型, 默认 :attr:`TargetType.CONTENT`.
            annotation_id: 已有 KPA 标注 ID, 为 None 时自动创建新标注;
                若指定但不存在, 则记录警告并创建新标注.
            trace_id: OpenTelemetry 全链路 trace ID.
            session_id: 会话 ID.

        Returns:
            桥接结果字典::

                {
                    "success": bool,           # 桥接是否成功
                    "annotation_id": str,      # KPA 标注 ID
                    "review_id": str,          # CC1 评审报告 ID
                    "verdict": str,            # CC3 verdict 字符串
                    "completeness": float,     # KPA 标注完整度 (0.0-1.0)
                    "bridge_event_id": str,    # 桥接事件 ID
                    "error": str,              # 错误信息 (失败时)
                }
        """
        start_time = time.time()
        self._stats["total_bridges"] += 1

        # --------------------------------------------------------
        # 输入校验
        # --------------------------------------------------------
        if review_result is None:
            latency_ms = (time.time() - start_time) * 1000.0
            self._stats["total_latency_ms_sum"] += latency_ms
            self._stats["failed_bridges"] += 1
            error_msg = "review_result 不能为 None"
            logger.error(error_msg)
            event = self._record_event(
                review_result=None,
                annotation_id=annotation_id or "",
                verdict="",
                scores={},
                cc3_result=None,
                success=False,
                error=error_msg,
                trace_id=trace_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )
            return {
                "success": False,
                "annotation_id": annotation_id or "",
                "review_id": "",
                "verdict": "",
                "completeness": 0.0,
                "bridge_event_id": event.event_id,
                "error": error_msg,
            }

        review_id = review_result.report_id

        # --------------------------------------------------------
        # Step 1: 确保 KPA 标注存在
        # --------------------------------------------------------
        try:
            original_annotation_id = annotation_id
            annotation_id = self._ensure_annotation(
                target_id=target_id,
                target_type=target_type,
                annotation_id=annotation_id,
            )
            # 新建标注统计
            if annotation_id != original_annotation_id:
                self._stats["annotations_created"] += 1
        except BridgeConnectionError as exc:
            latency_ms = (time.time() - start_time) * 1000.0
            self._stats["total_latency_ms_sum"] += latency_ms
            self._stats["failed_bridges"] += 1
            error_msg = str(exc)
            logger.exception(
                "CC1→CC3 标注准备失败: report_id=%s",
                review_id,
            )
            event = self._record_event(
                review_result=review_result,
                annotation_id=annotation_id or "",
                verdict="",
                scores={},
                cc3_result=None,
                success=False,
                error=error_msg,
                trace_id=trace_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )
            return {
                "success": False,
                "annotation_id": annotation_id or "",
                "review_id": review_id,
                "verdict": "",
                "completeness": 0.0,
                "bridge_event_id": event.event_id,
                "error": error_msg,
            }

        # --------------------------------------------------------
        # Step 2 ~ 4: 转换层评分 / 判决 / 提取问题与自纠次数
        # --------------------------------------------------------
        scores = self._convert_layer_scores(review_result.layer_scores)
        verdict = self._convert_verdict(review_result.verdict)
        issues = self._extract_issues(review_result)
        self_correction_count = self._count_self_corrections(review_result)

        # 统计: 按 CC3 verdict
        if verdict in self._stats["by_verdict"]:
            self._stats["by_verdict"][verdict] += 1

        # --------------------------------------------------------
        # Step 5: 经断路器调用 CC3 集成
        # --------------------------------------------------------
        cc3_result: dict[str, Any] | None = None
        error_msg = ""

        try:
            cc3_result = self._circuit_breaker.call(
                self._cc_integration.on_cc1_review_completed,
                annotation_id=annotation_id,
                review_id=review_id,
                scores=scores,
                verdict=verdict,
                issues=issues,
                self_correction_count=self_correction_count,
                trace_id=trace_id,
                session_id=session_id,
            )
        except CircuitBreakerOpenError as exc:
            error_msg = str(exc)
            self._stats["circuit_breaker_trips"] += 1
            logger.warning(
                "CC1→CC3 桥接断路器跳闸: report_id=%s, %s",
                review_id,
                exc,
            )
        except Exception as exc:
            bridge_error = BridgeConnectionError(
                source="cc1",
                target="cc3",
                reason=str(exc),
            )
            error_msg = str(bridge_error)
            logger.exception(
                "CC1→CC3 集成调用异常: report_id=%s",
                review_id,
            )

        # 判定桥接是否成功
        success = bool(
            cc3_result is not None and cc3_result.get("success", False)
        )

        if not success:
            # CC3 集成失败 — 记录降级事件并返回
            latency_ms = (time.time() - start_time) * 1000.0
            self._stats["total_latency_ms_sum"] += latency_ms
            self._stats["failed_bridges"] += 1
            if not error_msg:
                error_msg = cc3_result.get(
                    "error", "CC3 集成返回失败"
                ) if cc3_result else "CC3 集成返回失败"
            event = self._record_event(
                review_result=review_result,
                annotation_id=annotation_id,
                verdict=verdict,
                scores=scores,
                cc3_result=cc3_result,
                success=False,
                error=error_msg,
                trace_id=trace_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )
            return {
                "success": False,
                "annotation_id": annotation_id,
                "review_id": review_id,
                "verdict": verdict,
                "completeness": 0.0,
                "bridge_event_id": event.event_id,
                "error": error_msg,
            }

        # --------------------------------------------------------
        # Step 6: 统计 + 事件记录 + 返回
        # --------------------------------------------------------
        latency_ms = (time.time() - start_time) * 1000.0
        self._stats["total_latency_ms_sum"] += latency_ms
        self._stats["successful_bridges"] += 1

        event = self._record_event(
            review_result=review_result,
            annotation_id=annotation_id,
            verdict=verdict,
            scores=scores,
            cc3_result=cc3_result,
            success=True,
            error="",
            trace_id=trace_id,
            session_id=session_id,
            latency_ms=latency_ms,
        )

        return {
            "success": True,
            "annotation_id": annotation_id,
            "review_id": review_id,
            "verdict": verdict,
            "completeness": cc3_result.get("completeness", 0.0),
            "bridge_event_id": event.event_id,
            "error": "",
        }

    # ========================================================
    # 内部方法
    # ========================================================

    def _ensure_annotation(
        self,
        target_id: str,
        target_type: TargetType,
        annotation_id: str | None,
    ) -> str:
        """创建或复用 KPA 标注.

        - ``annotation_id`` 非空且标注存在 → 直接复用
        - ``annotation_id`` 为 None → 按 target_id / target_type 创建新标注
        - ``annotation_id`` 非空但标注不存在 → 记录警告并创建新标注

        Args:
            target_id: 标注对象 ID.
            target_type: 标注对象类型.
            annotation_id: 已有标注 ID (可为 None).

        Returns:
            KPA 标注 ID.

        Raises:
            BridgeConnectionError: 标注查询或创建失败.
        """
        # 复用已有标注
        if annotation_id:
            try:
                self._kpa_engine.get_annotation(annotation_id)
                logger.debug("复用 KPA 标注: %s", annotation_id)
                return annotation_id
            except Exception:
                logger.warning(
                    "指定的 KPA 标注 %s 不存在, 将创建新标注",
                    annotation_id,
                )

        # 创建新标注
        try:
            annotation = self._kpa_engine.create_annotation(
                target_type=target_type,
                target_id=target_id,
            )
            logger.info(
                "创建 KPA 标注: id=%s, target=%s/%s",
                annotation.annotation_id,
                target_type.value,
                target_id,
            )
            return annotation.annotation_id
        except Exception as exc:
            raise BridgeConnectionError(
                source="cc1",
                target="cc3",
                reason=f"KPA 标注创建失败: {exc}",
            ) from exc

    def _convert_layer_scores(
        self,
        layer_scores: dict[Any, float] | None,
    ) -> dict[str, float]:
        """将 ReviewLayerType 枚举键转换为 CC3 校验维度字符串键.

        映射关系:
            - ReviewLayerType.L1_FACT       → "factual"
            - ReviewLayerType.L2_LOGIC      → "logical"
            - ReviewLayerType.L3_NUMERICAL  → "numerical"
            - ReviewLayerType.L4_PROVENANCE → "provenance"

        兼容已为字符串键的输入 (按枚举值或原值映射, 幂等).

        Args:
            layer_scores: CC1 各层评分 (枚举键或字符串键).

        Returns:
            CC3 校验维度评分 (字符串键).
        """
        converted: dict[str, float] = {}
        for key, value in (layer_scores or {}).items():
            if isinstance(key, ReviewLayerType):
                str_key = LAYER_SCORE_KEY_MAP.get(key, key.value)
            else:
                # 字符串键: 优先按枚举值映射 (如 "l1_fact" → "factual"),
                # 已是目标键 (如 "factual") 则原样保留
                raw = getattr(key, "value", str(key))
                str_key = LAYER_SCORE_VALUE_MAP.get(raw, raw)
            converted[str_key] = float(value)
        return converted

    def _convert_verdict(self, verdict: Any) -> str:
        """将 CC1 ReviewVerdict 转换为 CC3 verdict 字符串.

        映射关系:
            - ReviewVerdict.PASS  → "pass"
            - ReviewVerdict.FLAG  → "pass_with_notes"
            - ReviewVerdict.BLOCK → "fail"

        未知 verdict 降级为 "pass" 并记录警告; 结果经
        :data:`VALID_CC3_VERDICTS` 校验, 非法值降级为 "pass".

        Args:
            verdict: CC1 评审判决 (ReviewVerdict 枚举或字符串).

        Returns:
            CC3 verdict 字符串 (pass / pass_with_notes / fail).
        """
        result: str | None = None

        if isinstance(verdict, ReviewVerdict):
            result = VERDICT_MAP.get(verdict)
        else:
            # 字符串兼容
            for k, v in VERDICT_MAP.items():
                if k.value == verdict or k == verdict:
                    result = v
                    break

        if result is None:
            logger.warning("未知 CC1 verdict: %r, 降级为 pass", verdict)
            result = "pass"

        # 校验为合法 CC3 verdict
        if result not in VALID_CC3_VERDICTS:
            logger.warning("非法 CC3 verdict: %r, 降级为 pass", result)
            result = ValidationVerdict.PASS.value

        return result

    def _extract_issues(
        self,
        review_result: ReviewResult,
    ) -> list[dict[str, Any]]:
        """从评审结果中提取问题列表.

        返回问题字典的浅拷贝列表, 避免修改原始评审结果;
        非字典元素包装为 ``{"value": str(issue)}``.

        Args:
            review_result: CC1 评审结果.

        Returns:
            问题列表 (每个问题为 dict).
        """
        issues = getattr(review_result, "issues", None) or []
        result: list[dict[str, Any]] = []
        for issue in issues:
            if isinstance(issue, dict):
                result.append(dict(issue))
            else:
                result.append({"value": str(issue)})
        return result

    def _count_self_corrections(
        self,
        review_result: ReviewResult,
    ) -> int:
        """统计自纠回路迭代次数.

        Args:
            review_result: CC1 评审结果.

        Returns:
            自纠回路已执行次数 (未触发自纠时为 0).
        """
        self_correction = getattr(review_result, "self_correction", None)
        if self_correction is None:
            return 0
        attempts = getattr(self_correction, "attempts", 0)
        try:
            return int(attempts)
        except (TypeError, ValueError):
            return 0

    def _record_event(
        self,
        review_result: ReviewResult | None,
        annotation_id: str,
        verdict: str,
        scores: dict[str, float],
        cc3_result: dict[str, Any] | None,
        success: bool,
        error: str,
        trace_id: str,
        session_id: str,
        latency_ms: float = 0.0,
    ) -> BridgeEvent:
        """记录桥接审计事件 (CloudEvents 格式).

        构建 :class:`BridgeEvent`, 包含 CC1 评审摘要、CC3 集成结果、
        转换后的层评分与 verdict.

        Args:
            review_result: CC1 评审结果 (可为 None).
            annotation_id: KPA 标注 ID.
            verdict: CC3 verdict 字符串.
            scores: 转换后的层评分.
            cc3_result: CC3 集成返回结果.
            success: 桥接是否成功.
            error: 错误信息 (成功时为空).
            trace_id: trace ID.
            session_id: 会话 ID.
            latency_ms: 桥接延迟 (毫秒).

        Returns:
            已记录的桥接事件.
        """
        # --- 构建事件负载 ---
        payload: dict[str, Any] = {
            "success": success,
            "error": error,
            "latency_ms": round(latency_ms, 2),
            "annotation_id": annotation_id,
            "verdict": verdict,
            "scores": scores,
        }

        if review_result is not None:
            payload["cc1"] = {
                "report_id": review_result.report_id,
                "verdict": (
                    review_result.verdict.value
                    if hasattr(review_result.verdict, "value")
                    else str(review_result.verdict)
                ),
                "composite_score": review_result.composite_score,
                "issues_count": len(review_result.issues),
                "self_correction_triggered": (
                    review_result.self_correction is not None
                ),
            }

        if cc3_result is not None:
            payload["cc3"] = {
                "success": cc3_result.get("success", False),
                "completeness": cc3_result.get("completeness", 0.0),
                "review_id": cc3_result.get("review_id", ""),
                "error": cc3_result.get("error", ""),
            }

        event = BridgeEvent(
            source="cc1",
            target="cc3",
            direction=BridgeDirection.CC1_TO_CC3,
            event_type="cc1_review_completed",
            trace_id=trace_id,
            session_id=session_id,
            payload=payload,
        )
        self._events.append(event)

        # 事件日志超限裁剪 (保留最近一半)
        if len(self._events) > self._MAX_EVENTS:
            keep = self._MAX_EVENTS // 2
            self._events = self._events[-keep:]

        return event

    # ========================================================
    # 统计与事件查询
    # ========================================================

    def get_statistics(self) -> dict[str, Any]:
        """返回桥接统计信息.

        Returns:
            统计字典, 包含::

                {
                    "total_bridges": int,         # 总桥接次数
                    "successful_bridges": int,    # 成功次数
                    "failed_bridges": int,        # 失败次数
                    "success_rate": float,        # 成功率 (0-100)
                    "circuit_breaker_trips": int, # 断路器跳闸次数
                    "annotations_created": int,   # 新建标注数
                    "by_verdict": dict,           # CC3 verdict 分布
                    "avg_latency_ms": float,      # 平均延迟 (毫秒)
                    "circuit_breaker_status": dict,
                }
        """
        total = self._stats["total_bridges"]
        successful = self._stats["successful_bridges"]

        avg_latency = (
            self._stats["total_latency_ms_sum"] / total if total > 0 else 0.0
        )
        success_rate = (successful / total * 100.0) if total > 0 else 0.0

        return {
            "total_bridges": total,
            "successful_bridges": successful,
            "failed_bridges": self._stats["failed_bridges"],
            "success_rate": round(success_rate, 2),
            "circuit_breaker_trips": self._stats["circuit_breaker_trips"],
            "annotations_created": self._stats["annotations_created"],
            "by_verdict": dict(self._stats["by_verdict"]),
            "avg_latency_ms": round(avg_latency, 2),
            "circuit_breaker_status": self._circuit_breaker.get_status(),
        }

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """返回最近的桥接事件.

        Args:
            limit: 返回事件数量上限 (默认 50).

        Returns:
            事件字典列表, 按时间倒序排列 (最新在前).
        """
        if limit <= 0:
            return []
        events = self._events[-limit:]
        events = list(reversed(events))  # 最新在前
        return [e.model_dump() for e in events]

    def reset(self) -> None:
        """重置桥接器状态.

        清空事件日志与统计计数器, 并重置断路器到 CLOSED 状态.
        """
        self._events.clear()
        self._stats = {
            "total_bridges": 0,
            "successful_bridges": 0,
            "failed_bridges": 0,
            "circuit_breaker_trips": 0,
            "annotations_created": 0,
            "by_verdict": {
                "pass": 0,
                "pass_with_notes": 0,
                "fail": 0,
            },
            "total_latency_ms_sum": 0.0,
        }
        self._circuit_breaker.reset()
        logger.info("CC1→CC3 桥接器已重置")


__all__ = ["CC1CC3Bridge"]
