"""G5 合规报告生成器 — NIST AI RMF 四函数映射.

融合 NIST AI RMF GOVERN/MAP/MEASURE/MANAGE + SOC2 合规框架映射：
- 四函数治理审计：治理、映射、度量、管理
- 合规框架自动映射（SOC2 / GDPR / 学术诚信）
- 生成结构化合规报告
- 风险评分与处置建议
"""

from __future__ import annotations

import enum
import time
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# NIST AI RMF 四函数模型
# ============================================================


class NISTFunction(str, enum.Enum):
    """NIST AI RMF 四函数."""

    GOVERN = "govern"
    MAP = "map"
    MEASURE = "measure"
    MANAGE = "manage"


class ComplianceControl(BaseModel):
    """合规控制点.

    单个合规控制项的评估结果。
    """

    control_id: str = Field(description="控制点 ID")
    name: str = Field(description="控制点名称")
    framework: str = Field(description="所属框架")
    nist_function: NISTFunction = Field(description="NIST 函数")
    description: str = Field(default="", description="描述")
    status: str = Field(default="compliant", description="状态")
    evidence: list[str] = Field(default_factory=list, description="证据列表")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="评分")
    findings: list[str] = Field(default_factory=list, description="发现项")
    recommendations: list[str] = Field(default_factory=list, description="建议")


class ComplianceDomain(BaseModel):
    """合规域.

    一组相关控制点的聚合评估。
    """

    domain_id: str = Field(description="域 ID")
    name: str = Field(description="域名称")
    controls: list[ComplianceControl] = Field(default_factory=list)
    overall_score: float = Field(default=1.0, ge=0.0, le=1.0)

    def compute_score(self) -> float:
        """计算域整体评分."""
        if not self.controls:
            return 1.0
        scores = [c.score for c in self.controls]
        return round(sum(scores) / len(scores), 3)


class GovernanceComplianceReport(BaseModel):
    """治理合规报告 (G5).

    完整的治理合规评估报告，基于 NIST AI RMF 四函数映射。
    """

    report_id: str = Field(
        default_factory=lambda: f"comp-{int(time.time())}",
    )
    title: str = Field(default="治理合规报告")
    generated_at: float = Field(default_factory=time.time)
    frameworks: list[str] = Field(default_factory=list)
    domains: list[ComplianceDomain] = Field(default_factory=list)
    overall_score: float = Field(default=1.0, ge=0.0, le=1.0)
    summary: str = Field(default="")
    risks: list[dict[str, Any]] = Field(default_factory=list)

    def compute_overall(self) -> float:
        """计算整体合规评分."""
        if not self.domains:
            return 1.0
        scores = [d.compute_score() for d in self.domains]
        return round(sum(scores) / len(scores), 3)


# ============================================================
# 合规报告生成器
# ============================================================


class ComplianceReporter:
    """合规报告生成器.

    基于审计数据和度量指标生成结构化合规报告。

    使用示例::

        reporter = ComplianceReporter()
        report = reporter.generate_from_audit(
            audit_stats={...},
            metrics_stats={...},
            frameworks=["SOC2", "NIST_AI_RMF"],
        )
        print(report.overall_score)
    """

    # SOC2 信任服务标准映射
    SOC2_CONTROLS: list[dict[str, Any]] = [
        {
            "control_id": "CC6.1",
            "name": "逻辑与物理访问控制",
            "description": "限制对系统组件的访问",
            "nist_function": NISTFunction.GOVERN,
        },
        {
            "control_id": "CC6.2",
            "name": "访问移除",
            "description": "及时移除终止人员访问权限",
            "nist_function": NISTFunction.MANAGE,
        },
        {
            "control_id": "CC7.1",
            "name": "系统操作监控",
            "description": "监控系统组件以检测异常",
            "nist_function": NISTFunction.MEASURE,
        },
        {
            "control_id": "CC7.2",
            "name": "系统操作异常处理",
            "description": "识别并处理系统操作异常",
            "nist_function": NISTFunction.MANAGE,
        },
        {
            "control_id": "A1.1",
            "name": "可用性监控",
            "description": "监控系统可用性",
            "nist_function": NISTFunction.MEASURE,
        },
    ]

    # NIST AI RMF 核心控制点
    NIST_CONTROLS: list[dict[str, Any]] = [
        {
            "control_id": "GOVERN-1",
            "name": "建立AI治理结构",
            "description": "定义AI治理的角色、责任和问责机制",
            "nist_function": NISTFunction.GOVERN,
        },
        {
            "control_id": "GOVERN-5",
            "name": "建立AI风险管理文化",
            "description": "在整个组织内培养风险管理意识",
            "nist_function": NISTFunction.GOVERN,
        },
        {
            "control_id": "MAP-1",
            "name": "AI系统上下文识别",
            "description": "识别AI系统的目的、利益相关者和影响范围",
            "nist_function": NISTFunction.MAP,
        },
        {
            "control_id": "MAP-5",
            "name": "AI系统影响评估",
            "description": "评估AI系统对个人、群体和社会的潜在影响",
            "nist_function": NISTFunction.MAP,
        },
        {
            "control_id": "MEASURE-1",
            "name": "AI系统性能评估",
            "description": "使用适当的指标评估AI系统性能",
            "nist_function": NISTFunction.MEASURE,
        },
        {
            "control_id": "MEASURE-2",
            "name": "AI系统鲁棒性评估",
            "description": "评估AI系统对异常输入的鲁棒性",
            "nist_function": NISTFunction.MEASURE,
        },
        {
            "control_id": "MANAGE-1",
            "name": "AI风险处置",
            "description": "根据风险评估结果处置风险",
            "nist_function": NISTFunction.MANAGE,
        },
        {
            "control_id": "MANAGE-4",
            "name": "AI事件响应",
            "description": "建立AI系统事件响应计划",
            "nist_function": NISTFunction.MANAGE,
        },
    ]

    # 学术诚信控制点
    ACADEMIC_CONTROLS: list[dict[str, Any]] = [
        {
            "control_id": "ACAD-1",
            "name": "防抄袭检测",
            "description": "检测和防止学术抄袭行为",
            "nist_function": NISTFunction.MEASURE,
        },
        {
            "control_id": "ACAD-2",
            "name": "学术内容溯源",
            "description": "确保学术内容的来源可追溯",
            "nist_function": NISTFunction.MAP,
        },
        {
            "control_id": "ACAD-3",
            "name": "评分公平性审计",
            "description": "审计评分过程的公平性",
            "nist_function": NISTFunction.MEASURE,
        },
        {
            "control_id": "ACAD-4",
            "name": "学术违规处置",
            "description": "建立学术违规处置流程",
            "nist_function": NISTFunction.MANAGE,
        },
    ]

    def __init__(self) -> None:
        self._framework_controls: dict[str, list[dict[str, Any]]] = {
            "SOC2": self.SOC2_CONTROLS,
            "NIST_AI_RMF": self.NIST_CONTROLS,
            "ACADEMIC_INTEGRITY": self.ACADEMIC_CONTROLS,
        }

    def generate_from_audit(
        self,
        audit_stats: dict[str, Any],
        metrics_stats: dict[str, Any] | None = None,
        frameworks: list[str] | None = None,
    ) -> GovernanceComplianceReport:
        """基于审计和度量数据生成合规报告.

        Args:
            audit_stats: 审计引擎统计
            metrics_stats: 度量引擎统计
            frameworks: 要评估的框架列表

        Returns:
            结构化合规报告
        """
        frameworks = frameworks or ["SOC2", "NIST_AI_RMF", "ACADEMIC_INTEGRITY"]
        metrics_stats = metrics_stats or {}

        domains: list[ComplianceDomain] = []
        risks: list[dict[str, Any]] = []

        for framework in frameworks:
            controls = self._framework_controls.get(framework, [])
            domain_controls: list[ComplianceControl] = []

            for ctrl_def in controls:
                control = self._evaluate_control(ctrl_def, audit_stats, metrics_stats)
                domain_controls.append(control)

                # 收集风险
                if control.score < 0.7:
                    risks.append({
                        "framework": framework,
                        "control_id": control.control_id,
                        "name": control.name,
                        "score": control.score,
                        "severity": "high" if control.score < 0.5 else "medium",
                        "findings": control.findings,
                    })

            domain = ComplianceDomain(
                domain_id=framework.lower(),
                name=framework,
                controls=domain_controls,
            )
            domain.overall_score = domain.compute_score()
            domains.append(domain)

        report = GovernanceComplianceReport(
            title="治理合规评估报告",
            frameworks=frameworks,
            domains=domains,
            risks=risks,
        )
        report.overall_score = report.compute_overall()
        report.summary = self._generate_summary(report)
        return report

    def _evaluate_control(
        self,
        ctrl_def: dict[str, Any],
        audit_stats: dict[str, Any],
        metrics_stats: dict[str, Any],
    ) -> ComplianceControl:
        """评估单个控制点."""
        control_id = ctrl_def["control_id"]
        nist_function = ctrl_def.get("nist_function", NISTFunction.GOVERN)

        score = 1.0
        findings: list[str] = []
        evidence: list[str] = []
        recommendations: list[str] = []

        # GOVERN 函数评估
        if nist_function == NISTFunction.GOVERN:
            evidence.append(f"审计日志总数: {audit_stats.get('total_recorded', 0)}")
            if audit_stats.get("unique_agents", 0) > 0:
                evidence.append(f"注册 Agent 数: {audit_stats['unique_agents']}")
            else:
                findings.append("未检测到 Agent 治理结构")
                score -= 0.3

        # MAP 函数评估
        elif nist_function == NISTFunction.MAP:
            if audit_stats.get("unique_traces", 0) > 0:
                evidence.append(f"追踪链数量: {audit_stats['unique_traces']}")
            else:
                findings.append("缺少分布式追踪上下文映射")
                score -= 0.2

        # MEASURE 函数评估
        elif nist_function == NISTFunction.MEASURE:
            slo_count = metrics_stats.get("registered_slos", 0)
            evidence.append(f"注册 SLO 数: {slo_count}")
            if slo_count == 0:
                findings.append("未定义 SLO 度量指标")
                score -= 0.4
            if audit_stats.get("total_anomalies", 0) > 0:
                evidence.append(f"异常检测数: {audit_stats['total_anomalies']}")
            if audit_stats.get("baselines", 0) == 0:
                findings.append("未建立行为基线")
                score -= 0.2

        # MANAGE 函数评估
        elif nist_function == NISTFunction.MANAGE:
            outcome_dist = audit_stats.get("outcome_distribution", {})
            total = sum(outcome_dist.values()) if outcome_dist else 0
            if total > 0:
                deny_count = outcome_dist.get("deny", 0) + outcome_dist.get("error", 0)
                error_rate = deny_count / total
                evidence.append(f"决策总数: {total}, 拒绝率: {error_rate:.2%}")
                if error_rate > 0.3:
                    findings.append(f"拒绝率过高 ({error_rate:.1%})，可能存在系统性问题")
                    score -= 0.3
            else:
                findings.append("无决策记录，无法评估处置效果")
                score -= 0.3

        score = max(0.0, min(1.0, score))

        # 生成建议
        if score < 0.5:
            recommendations.append(f"紧急改进: {ctrl_def['name']} 评分过低")
        elif score < 0.8:
            recommendations.append(f"建议优化: {ctrl_def['name']} 尚有提升空间")

        return ComplianceControl(
            control_id=control_id,
            name=ctrl_def["name"],
            framework=ctrl_def.get("framework", "GENERAL"),
            nist_function=nist_function,
            description=ctrl_def.get("description", ""),
            score=round(score, 3),
            status="compliant" if score >= 0.8 else "partial" if score >= 0.5 else "non_compliant",
            evidence=evidence,
            findings=findings,
            recommendations=recommendations,
        )

    def _generate_summary(self, report: GovernanceComplianceReport) -> str:
        """生成报告摘要."""
        score_pct = report.overall_score * 100
        risk_count = len(report.risks)

        if score_pct >= 90:
            status = "优秀"
        elif score_pct >= 80:
            status = "良好"
        elif score_pct >= 60:
            status = "合格"
        else:
            status = "需改进"

        domain_summaries = []
        for d in report.domains:
            domain_summaries.append(f"{d.name}: {d.overall_score*100:.1f}%")

        return (
            f"整体合规评分: {score_pct:.1f}% ({status}). "
            f"评估框架: {', '.join(report.frameworks)}. "
            f"发现风险: {risk_count} 项. "
            f"域评分: {'; '.join(domain_summaries)}."
        )

    def generate_nist_summary(self, report: GovernanceComplianceReport) -> dict[str, Any]:
        """生成 NIST 四函数摘要."""
        summary: dict[str, dict[str, Any]] = {
            "govern": {"controls": [], "avg_score": 0.0},
            "map": {"controls": [], "avg_score": 0.0},
            "measure": {"controls": [], "avg_score": 0.0},
            "manage": {"controls": [], "avg_score": 0.0},
        }

        for domain in report.domains:
            for ctrl in domain.controls:
                func = ctrl.nist_function.value
                summary[func]["controls"].append({
                    "id": ctrl.control_id,
                    "name": ctrl.name,
                    "score": ctrl.score,
                    "status": ctrl.status,
                })

        for func, data in summary.items():
            scores = [c["score"] for c in data["controls"]]
            data["avg_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0
            data["control_count"] = len(data["controls"])

        return summary
