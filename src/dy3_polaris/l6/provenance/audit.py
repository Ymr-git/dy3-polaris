"""溯源审计报告生成器.

生成结构化的审计报告，用于：
- 竞赛评审展示溯源完整性
- 教学场景下的知识来源追溯
- 异常诊断与责任定位
- 合规性检查

报告格式支持 JSON 导出和人类可读文本两种模式。
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.models import KPA, KPAEventType, LayerTag
from .chain import KPAChain
from .validator import ChainValidator, ValidationResult


class AuditReportGenerator:
    """溯源审计报告生成器.

    从 KPA 链生成多维度审计报告。

    使用示例:
        generator = AuditReportGenerator()
        report = generator.generate(chain)
        print(report["summary"])
        json_str = generator.to_json(report)
    """

    # 事件类型的中文描述映射
    EVENT_TYPE_LABELS: dict[str, str] = {
        "tool_invoked": "工具调用",
        "resource_read": "资源读取",
        "agent_output": "Agent 输出",
        "decision_routed": "决策路由",
        "human_override": "人工干预",
        "review_result": "审核结果",
        "knowledge_generated": "知识生成",
    }

    # 层标签中文描述
    LAYER_LABELS: dict[str, str] = {
        "L0": "L0 治理层",
        "L1": "L1 用户域",
        "L2": "L2 个性化",
        "L3": "L3 领域知识",
        "L4": "L4 决策引擎",
        "L5": "L5 Agent 运行时",
        "L6": "L6 协议层",
        "L7": "L7 体验层",
        "CC1": "CC1 防幻觉",
        "CC2": "CC2 人机协作",
        "CC3": "CC3 溯源层",
    }

    def __init__(self) -> None:
        self._validator = ChainValidator()

    def generate(self, chain: KPAChain) -> dict[str, Any]:
        """生成完整审计报告.

        Args:
            chain: KPA 链

        Returns:
            结构化审计报告字典
        """
        # 验证链完整性
        validation = self._validator.validate(chain, strict=False)

        # 基础信息
        report: dict[str, Any] = {
            "report_id": f"audit-{int(time.time())}",
            "generated_at": time.time(),
            "chain_info": {
                "chain_id": chain.chain_id,
                "length": chain.length,
                "sealed": chain.is_sealed,
                "created_at": chain.created_at,
                "head_hash": chain.head_hash,
                "genesis_hash": chain.genesis_hash,
                "duration_seconds": round(chain.duration_seconds(), 3),
            },
            "validation": validation.to_dict(),
            "summary": self._generate_summary(chain, validation),
            "event_timeline": self._generate_timeline(chain),
            "actor_analysis": self._generate_actor_analysis(chain),
            "layer_analysis": self._generate_layer_analysis(chain),
            "confidence_analysis": self._generate_confidence_analysis(chain),
            "context_refs": self._extract_context_refs(chain),
            "risk_assessment": self._assess_risks(chain, validation),
        }

        return report

    # --------------------------------------------------------
    # 报告各部分
    # --------------------------------------------------------

    def _generate_summary(self, chain: KPAChain, validation: ValidationResult) -> dict[str, Any]:
        """生成摘要."""
        event_counts = chain.event_type_counts()
        return {
            "total_events": chain.length,
            "is_valid": validation.is_valid,
            "error_count": validation.error_count,
            "warning_count": validation.warning_count,
            "event_types": {
                self.EVENT_TYPE_LABELS.get(k, k): v
                for k, v in event_counts.items()
            },
            "actors_involved": len(chain.actor_counts()),
            "layers_involved": len(chain.layer_counts()),
            "avg_confidence": round(chain.avg_confidence(), 4) if chain.avg_confidence() is not None else None,
            "duration_seconds": round(chain.duration_seconds(), 3),
            "is_sealed": chain.is_sealed,
        }

    def _generate_timeline(self, chain: KPAChain) -> list[dict[str, Any]]:
        """生成事件时间线."""
        timeline: list[dict[str, Any]] = []
        for i, kpa in enumerate(chain.kpas):
            timeline.append({
                "index": i,
                "kpa_id": kpa.kpa_id,
                "timestamp": kpa.timestamp,
                "event_type": kpa.event_type.value,
                "event_label": self.EVENT_TYPE_LABELS.get(kpa.event_type.value, kpa.event_type.value),
                "actor": kpa.actor,
                "layer": kpa.layer.value,
                "layer_label": self.LAYER_LABELS.get(kpa.layer.value, kpa.layer.value),
                "processing_logic": kpa.processing_logic,
                "confidence": kpa.confidence,
                "has_code_hash": kpa.code_hash is not None,
                "has_env_hash": kpa.env_hash is not None,
                "context_refs_count": len(kpa.context_refs),
                "prev_hash": kpa.prev_hash[:16] + "..." if kpa.prev_hash else None,
            })
        return timeline

    def _generate_actor_analysis(self, chain: KPAChain) -> list[dict[str, Any]]:
        """生成执行者分析."""
        actor_data: dict[str, dict[str, Any]] = {}
        for kpa in chain.kpas:
            if kpa.actor not in actor_data:
                actor_data[kpa.actor] = {
                    "actor": kpa.actor,
                    "total_events": 0,
                    "event_types": {},
                    "avg_confidence": [],
                    "layers": set(),
                    "first_seen": kpa.timestamp,
                    "last_seen": kpa.timestamp,
                }
            data = actor_data[kpa.actor]
            data["total_events"] += 1
            et = kpa.event_type.value
            data["event_types"][et] = data["event_types"].get(et, 0) + 1
            data["layers"].add(kpa.layer.value)
            if kpa.confidence is not None:
                data["avg_confidence"].append(kpa.confidence)
            data["last_seen"] = max(data["last_seen"], kpa.timestamp)
            data["first_seen"] = min(data["first_seen"], kpa.timestamp)

        result: list[dict[str, Any]] = []
        for actor, data in sorted(actor_data.items(), key=lambda x: -x[1]["total_events"]):
            confidences = data.pop("avg_confidence")
            layers = data.pop("layers")
            data["avg_confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else None
            data["layers"] = sorted(layers)
            data["active_duration_seconds"] = round(data["last_seen"] - data["first_seen"], 3)
            result.append(data)
        return result

    def _generate_layer_analysis(self, chain: KPAChain) -> list[dict[str, Any]]:
        """生成层标签分析."""
        layer_data: dict[str, dict[str, Any]] = {}
        for kpa in chain.kpas:
            layer = kpa.layer.value
            if layer not in layer_data:
                layer_data[layer] = {
                    "layer": layer,
                    "label": self.LAYER_LABELS.get(layer, layer),
                    "total_events": 0,
                    "actors": set(),
                    "event_types": {},
                }
            data = layer_data[layer]
            data["total_events"] += 1
            data["actors"].add(kpa.actor)
            et = kpa.event_type.value
            data["event_types"][et] = data["event_types"].get(et, 0) + 1

        result: list[dict[str, Any]] = []
        for layer, data in layer_data.items():
            data["actors"] = sorted(data.pop("actors"))
            result.append(data)
        return sorted(result, key=lambda x: x["layer"])

    def _generate_confidence_analysis(self, chain: KPAChain) -> dict[str, Any]:
        """生成置信度分析."""
        confidences = [kpa.confidence for kpa in chain.kpas if kpa.confidence is not None]
        if not confidences:
            return {
                "has_data": False,
                "count": 0,
                "avg": None,
                "min": None,
                "max": None,
                "low_confidence_count": 0,
            }

        return {
            "has_data": True,
            "count": len(confidences),
            "avg": round(sum(confidences) / len(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "low_confidence_count": sum(1 for c in confidences if c < 0.5),
        }

    def _extract_context_refs(self, chain: KPAChain) -> list[dict[str, Any]]:
        """提取所有上下文引用."""
        refs: list[dict[str, Any]] = []
        for kpa in chain.kpas:
            for ref in kpa.context_refs:
                refs.append({
                    "kpa_id": kpa.kpa_id,
                    "actor": kpa.actor,
                    "ref": ref,
                    "event_type": kpa.event_type.value,
                })
        return refs

    def _assess_risks(self, chain: KPAChain, validation: ValidationResult) -> dict[str, Any]:
        """风险评估."""
        risks: list[dict[str, str]] = []
        risk_level = "low"

        # 链验证失败
        if not validation.is_valid:
            risks.append({
                "risk": "chain_validation_failed",
                "level": "high",
                "description": f"链完整性验证失败，{validation.error_count} 个错误",
            })
            risk_level = "high"

        # 低置信度 KPA 过多
        conf_analysis = self._generate_confidence_analysis(chain)
        if conf_analysis["has_data"] and conf_analysis["count"] > 0:
            low_ratio = conf_analysis["low_confidence_count"] / conf_analysis["count"]
            if low_ratio > 0.3:
                risks.append({
                    "risk": "low_confidence_ratio",
                    "level": "medium",
                    "description": f"低置信度 KPA 占比 {low_ratio:.1%} ({conf_analysis['low_confidence_count']}/{conf_analysis['count']})",
                })
                if risk_level == "low":
                    risk_level = "medium"

        # 缺少 code_hash
        missing_code = sum(1 for kpa in chain.kpas if kpa.code_hash is None)
        if missing_code > 0 and chain.length > 0:
            missing_ratio = missing_code / chain.length
            if missing_ratio > 0.5:
                risks.append({
                    "risk": "missing_code_hash",
                    "level": "low",
                    "description": f"{missing_code}/{chain.length} 个 KPA 缺少 code_hash，影响可复现性",
                })

        # 未封存
        if not chain.is_sealed and chain.length > 0:
            risks.append({
                "risk": "chain_not_sealed",
                "level": "low",
                "description": "链未封存，存在被篡改风险",
            })

        # 时间戳异常
        if validation.warning_count > 0:
            for w in validation.warnings:
                if w.get("check") == "timestamp_order":
                    risks.append({
                        "risk": "timestamp_disorder",
                        "level": "low",
                        "description": "时间戳非单调递增，可能存在时钟同步问题",
                    })
                    break

        return {
            "level": risk_level,
            "risk_count": len(risks),
            "risks": risks,
        }

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def to_json(self, report: dict[str, Any], indent: int = 2) -> str:
        """将报告导出为 JSON 字符串."""
        return json.dumps(report, indent=indent, ensure_ascii=False, default=str)

    def to_text(self, report: dict[str, Any]) -> str:
        """将报告导出为人类可读文本."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("溯源审计报告")
        lines.append("=" * 60)
        lines.append("")

        # 摘要
        s = report["summary"]
        lines.append("【摘要】")
        lines.append(f"  总事件数: {s['total_events']}")
        lines.append(f"  链验证: {'通过' if s['is_valid'] else '失败'} (错误={s['error_count']}, 警告={s['warning_count']})")
        lines.append(f"  涉及执行者: {s['actors_involved']}")
        lines.append(f"  涉及层级: {s['layers_involved']}")
        lines.append(f"  平均置信度: {s['avg_confidence']}")
        lines.append(f"  持续时间: {s['duration_seconds']}s")
        lines.append(f"  已封存: {'是' if s['is_sealed'] else '否'}")
        lines.append("")

        # 事件类型分布
        lines.append("【事件类型分布】")
        for et, count in s["event_types"].items():
            lines.append(f"  {et}: {count}")
        lines.append("")

        # 时间线
        lines.append("【事件时间线】")
        for item in report["event_timeline"]:
            conf_str = f" conf={item['confidence']}" if item["confidence"] is not None else ""
            lines.append(
                f"  [{item['index']:3d}] {item['event_label']} | {item['actor']} | "
                f"{item['layer_label']}{conf_str}"
            )
        lines.append("")

        # 风险评估
        risk = report["risk_assessment"]
        lines.append(f"【风险评估】 等级: {risk['level']}")
        for r in risk["risks"]:
            lines.append(f"  [{r['level']}] {r['risk']}: {r['description']}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


__all__ = ["AuditReportGenerator"]
