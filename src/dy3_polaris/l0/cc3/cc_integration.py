"""CC3 溯源捕获层 — CC1/CC2 跨切面集成.

实现 CC3 与 CC1 (反幻觉) 和 CC2 (人机协作) 的双向集成:
- CC1 → CC3: 评审结果自动写入 KPA 校验维度
- CC2 → CC3: 审批/决策记录写入 KPA 决策维度
- CC3 → CC1: 溯源完整性反馈给 CC1 溯源层
- CC3 → CC2: 溯源缺失触发 CC2 审批

核心能力:
- CC1 评审结果到 KPA validation 维度的自动映射
- CC2 审批记录到 KPA decision 维度的自动映射
- 辩论触发条件的溯源检查
- 溯源缺失时的 CC2 升级建议
- 跨切面事件总线

融合方案:
- OpenTelemetry: trace_id 跨切面传递
- W3C PROV: 跨 Activity 关联映射
- Event-Driven Architecture: 松耦合事件驱动
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .kpa_engine import KPAEngine
from .debate_logger import DebateLogger
from .provenance_chain_builder import ProvenanceChainBuilder
from .ledger_integration import LedgerIntegration
from .models import (
    KPAAnnotation,
    TargetType,
    ValidationVerdict,
    CrossLayerDirection,
    EventType,
)

logger = logging.getLogger(__name__)


class CCIntegration:
    """CC1/CC2/CC3 跨切面集成器.

    提供 CC3 与其他横切机制的双向集成接口。

    使用示例::

        kpa = KPAEngine()
        dl = DebateLogger()
        chain = ProvenanceChainBuilder()
        ledger = LedgerIntegration()

        integration = CCIntegration(kpa, dl, chain, ledger)

        # CC1 评审完成后, 自动更新 KPA 校验维度
        integration.on_cc1_review_completed(
            annotation_id="kpa-001",
            review_id="rv-001",
            scores={"factual": 85, "logical": 90, "numerical": 88, "provenance": 82},
            verdict="pass",
        )

        # CC2 审批完成后, 自动更新 KPA 决策维度
        integration.on_cc2_approval_completed(
            annotation_id="kpa-001",
            approval_id="ap-001",
            approval_level="approval",
        )
    """

    def __init__(
        self,
        kpa_engine: KPAEngine | None = None,
        debate_logger: DebateLogger | None = None,
        chain_builder: ProvenanceChainBuilder | None = None,
        ledger: LedgerIntegration | None = None,
    ) -> None:
        """初始化跨切面集成器."""
        self._kpa = kpa_engine or KPAEngine()
        self._dl = debate_logger or DebateLogger()
        self._chain = chain_builder or ProvenanceChainBuilder()
        self._ledger = ledger or LedgerIntegration()

    # ==========================================================
    # CC1 → CC3: 评审结果写入
    # ==========================================================

    def on_cc1_review_completed(
        self,
        annotation_id: str,
        review_id: str,
        scores: dict[str, float],
        verdict: str = "pass",
        issues: list[dict[str, Any]] | None = None,
        self_correction_count: int = 0,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """CC1 评审完成后的回调 — 更新 KPA 校验维度.

        将 CC1 四层评审结果写入 KPA 标注的 validation 维度,
        并在溯源链中追加节点。

        Args:
            annotation_id: KPA 标注 ID
            review_id: CC1 评审报告 ID
            scores: 四层评分 {factual, logical, numerical, provenance}
            verdict: 评审结论 (pass/pass_with_notes/fail)
            issues: 问题列表
            self_correction_count: 自纠回路迭代次数
            trace_id: 全链路 trace ID
            session_id: 会话 ID

        Returns:
            更新结果
        """
        # 映射 verdict
        verdict_map = {
            "pass": ValidationVerdict.PASS,
            "pass_with_notes": ValidationVerdict.PASS_WITH_NOTES,
            "fail": ValidationVerdict.FAIL,
        }
        mapped_verdict = verdict_map.get(verdict, ValidationVerdict.PASS)

        # 更新 KPA 校验维度
        try:
            annotation = self._kpa.update_validation(
                annotation_id=annotation_id,
                cc1_review_id=review_id,
                four_layer_scores=scores,
                verdict=mapped_verdict,
                issues=issues,
                self_correction_count=self_correction_count,
            )
        except Exception as exc:
            logger.error("更新 KPA 校验维度失败: %s", exc)
            return {"success": False, "error": str(exc)}

        # 写入 Ledger
        self._ledger.write_kpa(
            annotation=annotation,
            trace_id=trace_id,
            session_id=session_id,
        )

        # 追加溯源链节点
        chains = self._chain.list_chains()
        if chains:
            chain_id = chains[0]["chain_id"]
            self._chain.append_node(
                chain_id=chain_id,
                annotation_id=annotation_id,
                target_id=annotation.target_id,
                agent_id="cc1-reviewer",
                agent_role="reviewer",
                layer="CC1",
                direction=CrossLayerDirection.CC1_TO_CC3,
            )

        # 写入跨层事件
        self._ledger.write_cross_layer(
            direction=CrossLayerDirection.CC1_TO_CC3,
            trace_id=trace_id,
            session_id=session_id,
            agent_id="cc1-reviewer",
            payload={
                "review_id": review_id,
                "annotation_id": annotation_id,
                "verdict": verdict,
                "scores": scores,
            },
        )

        logger.info(
            "CC1→CC3 集成: annotation=%s, review=%s, verdict=%s",
            annotation_id,
            review_id,
            verdict,
        )

        return {
            "success": True,
            "annotation_id": annotation_id,
            "review_id": review_id,
            "verdict": verdict,
            "completeness": annotation.completeness_score(),
        }

    # ==========================================================
    # CC2 → CC3: 审批结果写入
    # ==========================================================

    def on_cc2_approval_completed(
        self,
        annotation_id: str,
        approval_id: str,
        approval_level: str = "approval",
        meta_decider_result: str = "",
        paradigm_selected: str = "",
        debate_id: str = "",
        decision_path: list[str] | None = None,
        trace_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """CC2 审批完成后的回调 — 更新 KPA 决策维度.

        Args:
            annotation_id: KPA 标注 ID
            approval_id: CC2 审批记录 ID
            approval_level: CC2 协同层级
            meta_decider_result: Meta-Decider 决策结果
            paradigm_selected: 选择的讲解范式
            debate_id: 辩论 ID (如触发辩论)
            decision_path: 决策路径
            trace_id: 全链路 trace ID
            session_id: 会话 ID

        Returns:
            更新结果
        """
        try:
            annotation = self._kpa.update_decision(
                annotation_id=annotation_id,
                meta_decider_result=meta_decider_result,
                paradigm_selected=paradigm_selected,
                cc2_approval_id=approval_id,
                cc2_approval_level=approval_level,
                debate_id=debate_id,
                decision_path=decision_path,
            )
        except Exception as exc:
            logger.error("更新 KPA 决策维度失败: %s", exc)
            return {"success": False, "error": str(exc)}

        # 写入 Ledger
        self._ledger.write_kpa(
            annotation=annotation,
            trace_id=trace_id,
            session_id=session_id,
        )

        # 追加溯源链节点
        chains = self._chain.list_chains()
        if chains:
            chain_id = chains[0]["chain_id"]
            self._chain.append_node(
                chain_id=chain_id,
                annotation_id=annotation_id,
                target_id=annotation.target_id,
                agent_id="cc2-approval-system",
                agent_role="approver",
                layer="CC2",
                direction=CrossLayerDirection.CC2_TO_CC3,
            )

        # 写入跨层事件
        self._ledger.write_cross_layer(
            direction=CrossLayerDirection.CC2_TO_CC3,
            trace_id=trace_id,
            session_id=session_id,
            agent_id="cc2-approval-system",
            payload={
                "approval_id": approval_id,
                "annotation_id": annotation_id,
                "approval_level": approval_level,
                "paradigm": paradigm_selected,
            },
        )

        logger.info(
            "CC2→CC3 集成: annotation=%s, approval=%s, level=%s",
            annotation_id,
            approval_id,
            approval_level,
        )

        return {
            "success": True,
            "annotation_id": annotation_id,
            "approval_id": approval_id,
            "completeness": annotation.completeness_score(),
        }

    # ==========================================================
    # CC3 → CC1: 溯源完整性反馈
    # ==========================================================

    def check_provenance_for_cc1(
        self,
        annotation_id: str,
    ) -> dict[str, Any]:
        """为 CC1 提供溯源完整性检查.

        检查 KPA 标注的溯源维度是否完整,
        用于 CC1 L4 溯源层评分。

        Args:
            annotation_id: KPA 标注 ID

        Returns:
            溯源完整性报告::

                {
                    "annotation_id": str,
                    "source_complete": bool,
                    "source_tier": str,
                    "has_doi": bool,
                    "chain_verified": bool,
                    "completeness_score": float,
                    "recommendation": str,
                }
        """
        try:
            annotation = self._kpa.get_annotation(annotation_id)
        except Exception:
            return {
                "annotation_id": annotation_id,
                "source_complete": False,
                "recommendation": "标注不存在",
            }

        source = annotation.source
        source_complete = source.is_filled()
        has_doi = bool(source.primary_source and source.primary_source.startswith("10."))

        # 检查溯源链
        chain_verified = True
        chains = self._chain.list_chains()
        if chains:
            for chain_meta in chains:
                chain_id = chain_meta.get("chain_id", "")
                if chain_id:
                    report = self._chain.verify_chain(chain_id)
                    if not report["all_passed"]:
                        chain_verified = False
                        break

        completeness = annotation.completeness_score()

        # 推荐建议
        if not source_complete:
            recommendation = "来源维度不完整, 建议补充 DOI 或实验条件"
        elif not has_doi and source.source_type == "journal":
            recommendation = "期刊来源缺少 DOI, 建议添加"
        elif not chain_verified:
            recommendation = "溯源链存在断裂, 需要修复"
        elif completeness < 0.5:
            recommendation = "标注完整度较低, 建议补充更多维度"
        else:
            recommendation = "溯源完整"

        return {
            "annotation_id": annotation_id,
            "source_complete": source_complete,
            "source_tier": source.trust_tier.value,
            "has_doi": has_doi,
            "chain_verified": chain_verified,
            "completeness_score": round(completeness, 4),
            "recommendation": recommendation,
        }

    # ==========================================================
    # CC3 → CC2: 溯源缺失升级建议
    # ==========================================================

    def check_escalation_for_cc2(
        self,
        annotation_id: str,
    ) -> dict[str, Any]:
        """为 CC2 提供溯源缺失升级建议.

        当溯源不完整时, 建议 CC2 升级到更高级别的审批。

        Args:
            annotation_id: KPA 标注 ID

        Returns:
            升级建议::

                {
                    "annotation_id": str,
                    "needs_escalation": bool,
                    "reason": str,
                    "suggested_level": str,
                    "risk_factors": [...],
                }
        """
        prov_report = self.check_provenance_for_cc1(annotation_id)

        risk_factors: list[str] = []
        suggested_level = "implicit"
        needs_escalation = False
        reason = ""

        if not prov_report["source_complete"]:
            risk_factors.append("来源维度不完整")
            needs_escalation = True
            suggested_level = "approval"
            reason = "溯源来源缺失, 需要人工审批"

        if prov_report["source_tier"] in ("tier_4", "tier_5"):
            risk_factors.append(f"来源等级低: {prov_report['source_tier']}")
            needs_escalation = True
            if suggested_level == "implicit":
                suggested_level = "prompt"
                reason = "来源权威性不足, 需要确认"

        if not prov_report["chain_verified"]:
            risk_factors.append("溯源链不完整")
            needs_escalation = True
            suggested_level = "intervention"
            reason = "溯源链断裂, 需要人工干预"

        if prov_report["completeness_score"] < 0.3:
            risk_factors.append(f"完整度过低: {prov_report['completeness_score']:.2f}")
            needs_escalation = True
            if suggested_level in ("implicit", "prompt"):
                suggested_level = "approval"
                reason = "标注完整度严重不足"

        if not needs_escalation:
            reason = "溯源完整, 无需升级"
            suggested_level = "implicit"

        return {
            "annotation_id": annotation_id,
            "needs_escalation": needs_escalation,
            "reason": reason,
            "suggested_level": suggested_level,
            "risk_factors": risk_factors,
            "completeness_score": prov_report["completeness_score"],
        }

    # ==========================================================
    # 辩论触发溯源检查
    # ==========================================================

    def check_debate_trigger(
        self,
        annotation_id: str,
        complexity_score: float,
    ) -> dict[str, Any]:
        """检查是否应该触发辩论.

        当复杂度在 31-65 区间且溯源不完整时,
        建议触发辩论。

        Args:
            annotation_id: KPA 标注 ID
            complexity_score: 复杂度评分

        Returns:
            辩论触发建议
        """
        should_trigger = 31 <= complexity_score <= 65
        prov_report = self.check_provenance_for_cc1(annotation_id)

        if should_trigger and not prov_report["source_complete"]:
            return {
                "should_trigger": True,
                "reason": f"复杂度 {complexity_score} 在辩论区间且溯源不完整",
                "complexity_score": complexity_score,
                "source_complete": False,
                "focus_area": "溯源补充与验证",
            }

        if should_trigger:
            return {
                "should_trigger": True,
                "reason": f"复杂度 {complexity_score} 在辩论区间",
                "complexity_score": complexity_score,
                "source_complete": prov_report["source_complete"],
                "focus_area": "知识准确性辩论",
            }

        return {
            "should_trigger": False,
            "reason": f"复杂度 {complexity_score} 不在辩论区间 (31-65)",
            "complexity_score": complexity_score,
            "source_complete": prov_report["source_complete"],
        }


__all__ = ["CCIntegration"]
