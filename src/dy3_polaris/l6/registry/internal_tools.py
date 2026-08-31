"""11 个内部计算工具的 Schema 定义与 Stub 实现.

分类:
- 诊断工具 (3): bkt_compute, irt_evaluate, forgetfulness_scan
- 审查工具 (4): rule_engine_check, cross_validation, standard_value_check, fact_consistency
- 指导工具 (3): topology_analysis, path_simulation, resource_matching
- 共享工具 (1): literature_trace

每个工具包含完整的 input_schema / output_schema / Dy3ToolAnnotations。
命名规范: 内部工具使用下划线命名 (不用 internal. 前缀, 与 ToolRegistration pattern 一致)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from typing import Any

from ..core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)

logger = logging.getLogger(__name__)


# ============================================================
# 诊断工具 (3)
# ============================================================

def _bkt_compute_registration() -> ToolRegistration:
    """internal.bkt_compute — 贝叶斯知识追踪计算.

    基于四参数 BKT 模型 (P(T), P(S), P(G), P(L0)) 计算学习者掌握概率。
    """
    return ToolRegistration(
        name="bkt_compute",
        description="贝叶斯知识追踪(BKT)计算：基于学习者作答序列更新知识点掌握概率。四参数模型：P(L0)先验掌握率、P(T)转移概率、P(G)猜测率、P(S)失误率。",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {
                    "type": "string",
                    "description": "学习者唯一标识",
                },
                "kp_id": {
                    "type": "string",
                    "description": "知识点(KP)唯一标识，如 DOM-A-01",
                },
                "response": {
                    "type": "boolean",
                    "description": "本次作答是否正确",
                },
                "prior_p_know": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "先验掌握概率 P(L0)，默认 0.5",
                },
                "p_transit": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                    "description": "转移概率 P(T)：未掌握→掌握",
                },
                "p_guess": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.2,
                    "description": "猜测率 P(G)：未掌握但答对",
                },
                "p_slip": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                    "description": "失误率 P(S)：已掌握但答错",
                },
            },
            "required": ["learner_id", "kp_id", "response"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "p_know_posterior": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "后验掌握概率 P(L_t|response)",
                },
                "p_know_prior": {
                    "type": "number",
                    "description": "更新前掌握概率",
                },
                "mastery_threshold_met": {
                    "type": "boolean",
                    "description": "是否达到掌握阈值(>=0.95)",
                },
                "kp_id": {"type": "string"},
                "learner_id": {"type": "string"},
                "update_direction": {
                    "type": "string",
                    "enum": ["increase", "decrease"],
                    "description": "掌握概率变化方向",
                },
            },
            "required": ["p_know_posterior", "p_know_prior", "mastery_threshold_met"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["bkt", "diagnosis", "personalization", "L2", "knowledge_tracing"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=50,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=200,
        ),
    )


async def _bkt_compute_handler(
    learner_id: str,
    kp_id: str,
    response: bool,
    prior_p_know: float = 0.5,
    p_transit: float = 0.1,
    p_guess: float = 0.2,
    p_slip: float = 0.1,
) -> dict[str, Any]:
    """BKT 后验概率计算."""
    p_l = prior_p_know
    if response:
        # P(L|correct) = P(L) * (1 - P(S)) / (P(L) * (1 - P(S)) + (1 - P(L)) * P(G))
        numerator = p_l * (1 - p_slip)
        denominator = numerator + (1 - p_l) * p_guess
    else:
        # P(L|incorrect) = P(L) * P(S) / (P(L) * P(S) + (1 - P(L)) * (1 - P(G)))
        numerator = p_l * p_slip
        denominator = numerator + (1 - p_l) * (1 - p_guess)

    p_posterior = numerator / denominator if denominator > 0 else p_l

    # 加入转移概率
    p_posterior = p_posterior + (1 - p_posterior) * p_transit
    p_posterior = min(p_posterior, 0.9999)  # 防止溢出

    return {
        "p_know_posterior": round(p_posterior, 6),
        "p_know_prior": prior_p_know,
        "mastery_threshold_met": p_posterior >= 0.95,
        "kp_id": kp_id,
        "learner_id": learner_id,
        "update_direction": "increase" if p_posterior > prior_p_know else "decrease",
    }


def _irt_evaluate_registration() -> ToolRegistration:
    """internal.irt_evaluate — 项目反应理论(IRT)评估."""
    return ToolRegistration(
        name="irt_evaluate",
        description="项目反应理论(IRT)三参数模型评估：基于题目难度(b)、区分度(a)、猜测系数(c)和学习者能力(θ)计算正确作答概率。",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {"type": "string", "description": "学习者标识"},
                "theta": {
                    "type": "number",
                    "minimum": -3.0,
                    "maximum": 3.0,
                    "default": 0.0,
                    "description": "学习者能力参数 θ ∈ [-3, 3]",
                },
                "item_params": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "minimum": 0.0, "description": "区分度参数 a"},
                        "b": {"type": "number", "description": "难度参数 b"},
                        "c": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0, "description": "猜测系数 c"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                "item_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 1,
                    "description": "题目数量",
                },
            },
            "required": ["learner_id", "item_params"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "p_correct": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "正确作答概率 P(θ)",
                },
                "information": {
                    "type": "number",
                    "description": "题目信息函数 I(θ)",
                },
                "ability_estimate": {"type": "number", "description": "能力估计值"},
                "se_estimate": {"type": "number", "description": "标准误 SE"},
            },
            "required": ["p_correct", "information"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["irt", "diagnosis", "assessment", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=80,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=150,
        ),
    )


async def _irt_evaluate_handler(
    learner_id: str,
    theta: float = 0.0,
    item_params: dict | None = None,
    item_count: int = 1,
) -> dict[str, Any]:
    """IRT 三参数模型计算."""
    if item_params is None:
        item_params = {"a": 1.0, "b": 0.0, "c": 0.0}

    a = item_params.get("a", 1.0)
    b = item_params.get("b", 0.0)
    c = item_params.get("c", 0.0)

    # 三参数 logistic 模型: P(θ) = c + (1 - c) / (1 + exp(-1.7 * a * (θ - b)))
    z = -1.7 * a * (theta - b)
    p_correct = c + (1 - c) / (1 + math.exp(z))

    # 信息函数: I(θ) = a^2 * (1 - c)^2 * (P - c)^2 / ((1 - c)^2 * P * (1 - P))
    p = p_correct
    if p > c and p < 1:
        q = 1 - p
        numerator = a ** 2 * (p - c) ** 2
        denominator = (1 - c) ** 2 * p * q
        information = numerator / denominator if denominator > 0 else 0.0
    else:
        information = 0.0

    # 标准误
    se = 1.0 / math.sqrt(information) if information > 0 else 99.0

    return {
        "p_correct": round(p_correct, 6),
        "information": round(information, 6),
        "ability_estimate": theta,
        "se_estimate": round(se, 6),
    }


def _forgetfulness_scan_registration() -> ToolRegistration:
    """internal.forgetfulness_scan — 遗忘曲线扫描."""
    return ToolRegistration(
        name="forgetfulness_scan",
        description="遗忘曲线扫描：基于艾宾浩斯遗忘模型，扫描学习者所有已学知识点的遗忘状态，推荐复习时机。",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {"type": "string", "description": "学习者标识"},
                "kp_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kp_id": {"type": "string"},
                            "last_study_ts": {"type": "number", "description": "最后学习时间(Unix timestamp)"},
                            "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8, "description": "记忆强度"},
                        },
                        "required": ["kp_id", "last_study_ts"],
                    },
                    "minItems": 1,
                    "description": "需要扫描的知识点列表",
                },
                "current_ts": {
                    "type": "number",
                    "description": "当前时间戳(Unix)，默认取系统时间",
                },
                "forgetting_rate": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "遗忘速率参数(越大遗忘越快)",
                },
            },
            "required": ["learner_id", "kp_list"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "urgent_review": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kp_id": {"type": "string"},
                            "retention": {"type": "number"},
                            "recommended_review_in_hours": {"type": "number"},
                        },
                    },
                    "description": "需要紧急复习的知识点(保留率<0.5)",
                },
                "scheduled_review": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "计划复习的知识点(保留率0.5-0.8)",
                },
                "stable_kps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "稳定知识点(保留率>=0.8)",
                },
                "avg_retention": {"type": "number", "description": "平均保留率"},
            },
            "required": ["urgent_review", "scheduled_review", "stable_kps", "avg_retention"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["forgetting", "diagnosis", "ebbinghaus", "review", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=100,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _forgetfulness_scan_handler(
    learner_id: str,
    kp_list: list[dict],
    current_ts: float | None = None,
    forgetting_rate: float = 0.3,
) -> dict[str, Any]:
    """遗忘曲线扫描."""
    if current_ts is None:
        import time
        current_ts = time.time()

    urgent: list[dict] = []
    scheduled: list[dict] = []
    stable: list[str] = []
    total_retention = 0.0

    for kp in kp_list:
        kp_id = kp["kp_id"]
        last_ts = kp["last_study_ts"]
        strength = kp.get("strength", 0.8)

        hours_elapsed = (current_ts - last_ts) / 3600.0
        # R = exp(-t/S)，其中 S = strength / forgetting_rate
        s = strength / forgetting_rate
        retention = math.exp(-hours_elapsed / s) if s > 0 else 0.0
        retention = max(0.0, min(1.0, retention))
        total_retention += retention

        review_in = -s * math.log(0.5) / 24.0 if retention > 0 else 0  # 衰减到50%还需多少天

        entry = {
            "kp_id": kp_id,
            "retention": round(retention, 4),
            "recommended_review_in_hours": round(max(0, review_in * 24), 1),
        }

        if retention < 0.5:
            urgent.append(entry)
        elif retention < 0.8:
            scheduled.append(entry)
        else:
            stable.append(kp_id)

    avg = total_retention / len(kp_list) if kp_list else 0.0

    return {
        "urgent_review": urgent,
        "scheduled_review": scheduled,
        "stable_kps": stable,
        "avg_retention": round(avg, 4),
    }


# ============================================================
# 审查工具 (4)
# ============================================================

def _rule_engine_check_registration() -> ToolRegistration:
    """internal.rule_engine_check — 规则引擎审查."""
    return ToolRegistration(
        name="rule_engine_check",
        description="规则引擎审查：基于预定义教学规则集，对Agent输出的教学策略进行合规性检查，识别违规或偏离规则的行为。",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "待审查的内容文本"},
                "rule_set_id": {
                    "type": "string",
                    "default": "default_v1",
                    "description": "规则集标识",
                },
                "agent_id": {
                    "type": "string",
                    "description": "产出此内容的 Agent ID",
                },
                "context": {
                    "type": "object",
                    "description": "上下文信息(领域、KP等)",
                    "properties": {
                        "domain": {"type": "string"},
                        "kp_id": {"type": "string"},
                    },
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "passed": {"type": "boolean", "description": "是否通过全部规则"},
                "violations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string"},
                            "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                            "message": {"type": "string"},
                            "position": {"type": "object"},
                        },
                    },
                    "description": "违规列表",
                },
                "rule_count": {"type": "integer", "description": "检查的规则总数"},
                "compliance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["passed", "violations", "rule_count", "compliance_score"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["rule_engine", "review", "compliance", "CC1", "anti_hallucination"],
            layer=LayerTag.CC1_ANTI_HALLUCINATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=120,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _rule_engine_check_handler(
    content: str,
    rule_set_id: str = "default_v1",
    agent_id: str = "",
    context: dict | None = None,
) -> dict[str, Any]:
    """规则引擎审查 (stub: 基本长度和关键词检查)."""
    violations: list[dict] = []
    rule_count = 5

    # 规则 1: 内容不能为空
    if len(content.strip()) < 10:
        violations.append({
            "rule_id": "R001",
            "severity": "error",
            "message": "Content too short (minimum 10 characters)",
        })

    # 规则 2: 检查是否包含绝对化用语
    absolute_words = ["必须", "一定", "绝对", "never", "always", "impossible"]
    for word in absolute_words:
        if word.lower() in content.lower():
            violations.append({
                "rule_id": "R002",
                "severity": "warning",
                "message": f"Absolute term detected: '{word}'",
            })

    # 规则 3: 检查内容长度上限
    if len(content) > 10000:
        violations.append({
            "rule_id": "R003",
            "severity": "warning",
            "message": "Content exceeds 10000 characters",
        })

    passed = len(violations) == 0
    error_count = sum(1 for v in violations if v["severity"] == "error")
    compliance_score = max(0.0, 1.0 - error_count * 0.3 - (len(violations) - error_count) * 0.1)

    return {
        "passed": passed,
        "violations": violations,
        "rule_count": rule_count,
        "compliance_score": round(compliance_score, 4),
    }


def _cross_validation_registration() -> ToolRegistration:
    """internal.cross_validation — 交叉验证."""
    return ToolRegistration(
        name="cross_validation",
        description="交叉验证：对多个Agent的独立输出进行交叉验证，检测一致性和分歧，用于冲突检测和置信度校准。",
        input_schema={
            "type": "object",
            "properties": {
                "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string"},
                            "result": {"type": "string", "description": "Agent输出的文本结果"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["agent_id", "result"],
                    },
                    "minItems": 2,
                    "description": "至少两个Agent的输出",
                },
                "validation_method": {
                    "type": "string",
                    "enum": ["semantic", "lexical", "hybrid"],
                    "default": "hybrid",
                    "description": "验证方法：语义/词法/混合",
                },
            },
            "required": ["outputs"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "consensus": {"type": "boolean", "description": "是否达成共识"},
                "agreement_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "一致性分数"},
                "conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_ids": {"type": "array", "items": {"type": "string"}},
                            "issue": {"type": "string"},
                        },
                    },
                    "description": "检测到的冲突",
                },
                "recommended_confidence": {"type": "number", "description": "校准后的置信度"},
            },
            "required": ["consensus", "agreement_score", "conflicts"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["cross_validation", "review", "consensus", "CC1"],
            layer=LayerTag.CC1_ANTI_HALLUCINATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=200,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _cross_validation_handler(
    outputs: list[dict],
    validation_method: str = "hybrid",
) -> dict[str, Any]:
    """交叉验证 (stub: 基于 Jaccard 相似度)."""
    if len(outputs) < 2:
        return {
            "consensus": True,
            "agreement_score": 1.0,
            "conflicts": [],
            "recommended_confidence": outputs[0].get("confidence", 0.8) if outputs else 0.5,
        }

    # 词法相似度 (Jaccard)
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    token_sets = [_tokenize(o["result"]) for o in outputs]
    pairwise_scores: list[float] = []

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if union:
                score = len(token_sets[i] & token_sets[j]) / len(union)
            else:
                score = 1.0
            pairwise_scores.append(score)

    avg_score = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 1.0

    conflicts: list[dict] = []
    if avg_score < 0.5:
        conflicts.append({
            "agent_ids": [o["agent_id"] for o in outputs],
            "issue": f"Low agreement score ({avg_score:.2f}): outputs diverge significantly",
        })

    # 校准置信度
    confidences = [o.get("confidence", 0.8) for o in outputs]
    avg_conf = sum(confidences) / len(confidences)
    recommended_confidence = avg_conf * (0.5 + 0.5 * avg_score)  # 一致性折扣

    return {
        "consensus": avg_score >= 0.7,
        "agreement_score": round(avg_score, 4),
        "conflicts": conflicts,
        "recommended_confidence": round(recommended_confidence, 4),
    }


def _standard_value_check_registration() -> ToolRegistration:
    """internal.standard_value_check — 标准值校验."""
    return ToolRegistration(
        name="standard_value_check",
        description="标准值校验：将Agent输出的数值/结论与标准知识库中的参考值进行比对，检测偏差和错误。",
        input_schema={
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kp_id": {"type": "string", "description": "知识点ID"},
                            "field": {"type": "string", "description": "字段名，如 'boiling_point'"},
                            "value": {"description": "待校验的值(number/string)"},
                            "unit": {"type": "string", "description": "单位"},
                        },
                        "required": ["kp_id", "field", "value"],
                    },
                    "minItems": 1,
                    "description": "待校验的声明列表",
                },
                "tolerance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.05,
                    "description": "容差比例(5%)",
                },
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "all_valid": {"type": "boolean"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kp_id": {"type": "string"},
                            "field": {"type": "string"},
                            "input_value": {},
                            "standard_value": {},
                            "deviation": {"type": "number"},
                            "is_valid": {"type": "boolean"},
                            "severity": {"type": "string", "enum": ["ok", "minor", "major", "critical"]},
                        },
                    },
                },
                "summary": {"type": "object"},
            },
            "required": ["all_valid", "results"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["standard_check", "review", "fact_check", "CC1"],
            layer=LayerTag.CC1_ANTI_HALLUCINATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=150,
            domain_scope=["DOM-A", "DOM-B"],
            rate_limit=100,
        ),
    )


async def _standard_value_check_handler(
    claims: list[dict],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """标准值校验 (stub: 模拟标准库查询)."""
    # 模拟标准值库
    mock_standards: dict[str, dict[str, Any]] = {
        "DOM-A-01": {"boiling_point": 100.0, "melting_point": 0.0, "density": 1.0},
        "DOM-A-02": {"boiling_point": 78.3, "melting_point": -114.1, "density": 0.789},
    }

    results: list[dict] = []
    all_valid = True

    for claim in claims:
        kp_id = claim["kp_id"]
        field = claim["field"]
        value = claim["value"]

        standard = mock_standards.get(kp_id, {}).get(field)

        if standard is None:
            results.append({
                "kp_id": kp_id,
                "field": field,
                "input_value": value,
                "standard_value": None,
                "deviation": 0.0,
                "is_valid": True,
                "severity": "ok",
            })
            continue

        if isinstance(value, (int, float)) and isinstance(standard, (int, float)):
            if standard != 0:
                deviation = abs(value - standard) / abs(standard)
            else:
                deviation = abs(value - standard)

            is_valid = deviation <= tolerance
            if not is_valid:
                all_valid = False

            if deviation <= tolerance:
                severity = "ok"
            elif deviation <= tolerance * 3:
                severity = "minor"
            elif deviation <= tolerance * 10:
                severity = "major"
            else:
                severity = "critical"

            results.append({
                "kp_id": kp_id,
                "field": field,
                "input_value": value,
                "standard_value": standard,
                "deviation": round(deviation, 6),
                "is_valid": is_valid,
                "severity": severity,
            })
        else:
            is_valid = str(value) == str(standard)
            if not is_valid:
                all_valid = False
            results.append({
                "kp_id": kp_id,
                "field": field,
                "input_value": value,
                "standard_value": standard,
                "deviation": 0.0 if is_valid else 1.0,
                "is_valid": is_valid,
                "severity": "ok" if is_valid else "major",
            })

    valid_count = sum(1 for r in results if r["is_valid"])
    return {
        "all_valid": all_valid,
        "results": results,
        "summary": {
            "total": len(results),
            "valid": valid_count,
            "invalid": len(results) - valid_count,
        },
    }


def _fact_consistency_registration() -> ToolRegistration:
    """internal.fact_consistency — 事实一致性检查."""
    return ToolRegistration(
        name="fact_consistency",
        description="事实一致性检查：验证Agent生成内容中事实陈述的内部一致性，检测自相矛盾的论述。",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "待检查的文本内容"},
                "known_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "statement": {"type": "string"},
                            "verified": {"type": "boolean", "default": True},
                        },
                        "required": ["statement"],
                    },
                    "description": "已知事实列表(用于交叉验证)",
                },
                "domain": {"type": "string", "description": "领域标识"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "is_consistent": {"type": "boolean"},
                "inconsistencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["internal", "external"]},
                            "description": {"type": "string"},
                            "segment": {"type": "string"},
                        },
                    },
                },
                "fact_coverage": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "事实覆盖比率"},
                "confidence": {"type": "number"},
            },
            "required": ["is_consistent", "inconsistencies", "fact_coverage"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["fact_check", "consistency", "review", "CC1"],
            layer=LayerTag.CC1_ANTI_HALLUCINATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=180,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=80,
        ),
    )


async def _fact_consistency_handler(
    content: str,
    known_facts: list[dict] | None = None,
    domain: str = "",
) -> dict[str, Any]:
    """事实一致性检查 (stub)."""
    if known_facts is None:
        known_facts = []

    inconsistencies: list[dict] = []

    # 简单检查：数字一致性
    import re
    numbers = re.findall(r"(\d+\.?\d*)\s*(℃|°C|K|g/cm³|kg/m³|mol/L|kJ/mol|eV|nm|μm|mm)", content)
    seen_values: dict[str, list[str]] = {}
    for val, unit in numbers:
        if unit not in seen_values:
            seen_values[unit] = []
        seen_values[unit].append(val)

    for unit, vals in seen_values.items():
        unique_vals = set(vals)
        if len(unique_vals) > 1:
            inconsistencies.append({
                "type": "internal",
                "description": f"Multiple different values for unit '{unit}': {', '.join(unique_vals)}",
                "segment": f"Values: {vals}",
            })

    # 事实覆盖
    if known_facts:
        covered = sum(1 for f in known_facts if f["statement"].lower() in content.lower())
        coverage = covered / len(known_facts)
    else:
        coverage = 1.0

    is_consistent = len(inconsistencies) == 0
    confidence = max(0.0, 1.0 - len(inconsistencies) * 0.2)

    return {
        "is_consistent": is_consistent,
        "inconsistencies": inconsistencies,
        "fact_coverage": round(coverage, 4),
        "confidence": round(confidence, 4),
    }


# ============================================================
# 指导工具 (3)
# ============================================================

def _topology_analysis_registration() -> ToolRegistration:
    """internal.topology_analysis — 知识拓扑分析."""
    return ToolRegistration(
        name="topology_analysis",
        description="知识拓扑分析：分析知识点之间的前置/后继/并列关系，构建知识拓扑图，用于学习路径规划。",
        input_schema={
            "type": "object",
            "properties": {
                "kp_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "知识点ID列表",
                },
                "domain": {"type": "string", "description": "领域标识"},
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 2,
                    "description": "分析深度(展开层数)",
                },
            },
            "required": ["kp_ids"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kp_id": {"type": "string"},
                            "level": {"type": "integer"},
                            "in_degree": {"type": "integer"},
                            "out_degree": {"type": "integer"},
                        },
                    },
                    "description": "拓扑节点",
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "relation": {"type": "string", "enum": ["prerequisite", "corequisite", "successor"]},
                        },
                    },
                    "description": "拓扑边",
                },
                "entry_points": {"type": "array", "items": {"type": "string"}, "description": "入度为0的起点"},
                "exit_points": {"type": "array", "items": {"type": "string"}, "description": "出度为0的终点"},
                "topological_order": {"type": "array", "items": {"type": "string"}, "description": "拓扑排序"},
            },
            "required": ["nodes", "edges", "entry_points", "exit_points"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["topology", "knowledge_graph", "guidance", "L4"],
            layer=LayerTag.L4_DECISION_ENGINE,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=150,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _topology_analysis_handler(
    kp_ids: list[str],
    domain: str = "",
    depth: int = 2,
) -> dict[str, Any]:
    """知识拓扑分析 (stub: 构建简单链式拓扑)."""
    # 模拟知识图谱：链式+少量分支
    mock_edges: list[dict] = []
    for i in range(len(kp_ids) - 1):
        mock_edges.append({
            "from": kp_ids[i],
            "to": kp_ids[i + 1],
            "relation": "prerequisite",
        })

    # 计算入度和出度
    in_degree: dict[str, int] = {kp: 0 for kp in kp_ids}
    out_degree: dict[str, int] = {kp: 0 for kp in kp_ids}
    for edge in mock_edges:
        out_degree[edge["from"]] = out_degree.get(edge["from"], 0) + 1
        in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1

    nodes = [
        {
            "kp_id": kp,
            "level": 0,
            "in_degree": in_degree.get(kp, 0),
            "out_degree": out_degree.get(kp, 0),
        }
        for kp in kp_ids
    ]

    entry_points = [kp for kp in kp_ids if in_degree.get(kp, 0) == 0]
    exit_points = [kp for kp in kp_ids if out_degree.get(kp, 0) == 0]

    return {
        "nodes": nodes,
        "edges": mock_edges,
        "entry_points": entry_points,
        "exit_points": exit_points,
        "topological_order": kp_ids,  # 链式已排序
    }


def _path_simulation_registration() -> ToolRegistration:
    """internal.path_simulation — 学习路径模拟."""
    return ToolRegistration(
        name="path_simulation",
        description="学习路径模拟：基于知识拓扑和学习者当前状态，模拟不同学习路径的效果，推荐最优路径。",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {"type": "string"},
                "start_kp": {"type": "string", "description": "起始知识点"},
                "target_kp": {"type": "string", "description": "目标知识点"},
                "current_state": {
                    "type": "object",
                    "description": "当前知识状态",
                    "properties": {
                        "mastered_kps": {"type": "array", "items": {"type": "string"}},
                        "weak_kps": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_steps": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                        "preferred_difficulty": {"type": "string", "enum": ["easy", "medium", "hard"], "default": "medium"},
                        "time_budget_minutes": {"type": "integer", "default": 60},
                    },
                },
            },
            "required": ["learner_id", "start_kp", "target_kp"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "recommended_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "推荐学习路径(KP ID序列)",
                },
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "array", "items": {"type": "string"}},
                            "estimated_time_minutes": {"type": "integer"},
                            "difficulty": {"type": "string"},
                            "success_probability": {"type": "number"},
                        },
                    },
                    "description": "备选路径",
                },
                "estimated_time_minutes": {"type": "integer"},
                "success_probability": {"type": "number"},
                "rationale": {"type": "string", "description": "推荐理由"},
            },
            "required": ["recommended_path", "estimated_time_minutes", "success_probability"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["path_simulation", "learning_path", "guidance", "L4"],
            layer=LayerTag.L4_DECISION_ENGINE,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=300,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
            requires_compute=True,
        ),
    )


def _path_simulation_handler(
    learner_id: str,
    start_kp: str,
    target_kp: str,
    current_state: dict | None = None,
    constraints: dict | None = None,
) -> dict[str, Any]:
    """学习路径模拟 — 委托 L4 唯一策略决策点 (next-action).

    策略归位 L4: 由 l4.learning_strategy.generate_next_action 生成
    recommended_path (统一 {kp_id, action, target, effort} 结构);
    保留 recommended_path 字符串数组兼容旧调用方 (deprecated).
    """
    if current_state is None:
        current_state = {}
    if constraints is None:
        constraints = {}

    max_steps = min(int(constraints.get("max_steps", 10)), 50)
    difficulty = constraints.get("preferred_difficulty", "medium")

    # L4 唯一策略决策: 基于画像薄弱点生成路径 (无画像时由 start/target 拼装)
    from dy3_polaris.l4.learning_strategy import generate_next_action

    profile = {
        "kp_mastery": current_state.get("mastered_kps") or {},
        "weak_kps": current_state.get("weak_kps") or [],
    }
    decision = generate_next_action(profile, mode="guide")
    steps = decision["recommended_path"][:max_steps]

    if steps:
        path = [st["kp_id"] for st in steps]
        rationale = decision["summary"]
    else:
        # 无画像/无薄弱点: 兼容旧语义 (起点→终点)
        path = [start_kp, target_kp]
        rationale = (
            f"Path optimized for {difficulty} difficulty, balancing coverage and efficiency."
        )

    est_time = len(path) * 15  # 每步15分钟
    success_prob = 0.75 + random.uniform(-0.1, 0.15)

    alternatives = [
        {
            "path": [start_kp, target_kp],
            "estimated_time_minutes": 30,
            "difficulty": "hard",
            "success_probability": round(max(0.3, success_prob - 0.3), 4),
        }
    ]

    return {
        "recommended_path": path,
        "recommended_path_detail": steps,  # L4 统一语义: {kp_id, action, target, effort}
        "alternatives": alternatives,
        "estimated_time_minutes": est_time,
        "success_probability": round(success_prob, 4),
        "action_type": decision["action_type"],
        "confidence": decision["confidence"],
        "decision_source": "l4.next_action",
        "rationale": rationale,
    }


def _resource_matching_registration() -> ToolRegistration:
    """internal.resource_matching — 学习资源匹配."""
    return ToolRegistration(
        name="resource_matching",
        description="学习资源匹配：基于学习者画像和知识点需求，从资源库中匹配最合适的学习资源(视频/文档/习题/实验)。",
        input_schema={
            "type": "object",
            "properties": {
                "learner_id": {"type": "string"},
                "kp_id": {"type": "string", "description": "目标知识点"},
                "learner_style": {
                    "type": "string",
                    "enum": ["visual", "auditory", "reading", "kinesthetic"],
                    "default": "visual",
                    "description": "学习风格偏好",
                },
                "current_level": {
                    "type": "string",
                    "enum": ["novice", "intermediate", "advanced"],
                    "default": "novice",
                },
                "resource_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["video", "document", "exercise", "experiment", "simulation"]},
                    "default": ["video", "document"],
                    "description": "期望的资源类型",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["learner_id", "kp_id"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "matched_resources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string"},
                            "title": {"type": "string"},
                            "type": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "difficulty_match": {"type": "boolean"},
                            "estimated_time_minutes": {"type": "integer"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "best_match": {"type": "string", "description": "最佳匹配资源ID"},
            },
            "required": ["matched_resources", "total_found"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["resource_matching", "personalization", "guidance", "L2", "L4"],
            layer=LayerTag.L4_DECISION_ENGINE,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=120,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _resource_matching_handler(
    learner_id: str,
    kp_id: str,
    learner_style: str = "visual",
    current_level: str = "novice",
    resource_types: list[str] | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """学习资源匹配 (stub)."""
    if resource_types is None:
        resource_types = ["video", "document"]

    # 模拟资源库
    mock_resources = [
        {"resource_id": f"RES-{kp_id}-V01", "title": f"{kp_id} 入门视频", "type": "video", "base_relevance": 0.9},
        {"resource_id": f"RES-{kp_id}-D01", "title": f"{kp_id} 核心文档", "type": "document", "base_relevance": 0.85},
        {"resource_id": f"RES-{kp_id}-E01", "title": f"{kp_id} 练习题", "type": "exercise", "base_relevance": 0.75},
        {"resource_id": f"RES-{kp_id}-S01", "title": f"{kp_id} 仿真实验", "type": "simulation", "base_relevance": 0.7},
    ]

    # 过滤类型 + 计算匹配分
    matched = []
    for res in mock_resources:
        if res["type"] not in resource_types:
            continue

        # 学习风格加成
        style_bonus = 0.1 if learner_style == "visual" and res["type"] == "video" else 0.0
        level_bonus = 0.05 if current_level == "novice" and "入门" in res["title"] else 0.0
        relevance = min(1.0, res["base_relevance"] + style_bonus + level_bonus)

        matched.append({
            "resource_id": res["resource_id"],
            "title": res["title"],
            "type": res["type"],
            "relevance_score": round(relevance, 4),
            "difficulty_match": True,
            "estimated_time_minutes": random.randint(10, 45),
        })

    matched.sort(key=lambda x: x["relevance_score"], reverse=True)
    matched = matched[:max_results]

    return {
        "matched_resources": matched,
        "total_found": len(matched),
        "best_match": matched[0]["resource_id"] if matched else None,
    }


# ============================================================
# 画布生成工具 (1): 知识生成 Agent 的视觉化能力
# ============================================================

def _canvas_generation_registration() -> ToolRegistration:
    """internal.canvas_generation — 画布生成工具.

    为知识生成 Agent 提供画布能力，根据需求生成架构图、流程图、
    思维导图、时间线、泳道图和视觉化总结。输出 Mermaid.js 代码，
    前端使用 Mermaid 渲染器渲染。
    """
    return ToolRegistration(
        name="canvas_generation",
        description=(
            "画布生成工具：为知识生成 Agent 提供视觉化能力，"
            "根据内容生成 Mermaid.js 格式的架构图、流程图、思维导图、"
            "时间线、泳道图和视觉化总结。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "diagram_type": {
                    "type": "string",
                    "enum": [
                        "flowchart",       # 流程图
                        "mindmap",         # 思维导图
                        "timeline",        # 时间线
                        "swimlane",        # 泳道图
                        "architecture",    # 架构图
                        "visual_summary",  # 视觉化总结
                    ],
                    "description": "画布/图表类型",
                },
                "title": {
                    "type": "string",
                    "description": "图表标题",
                },
                "content": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "节点标识"},
                                    "label": {"type": "string", "description": "节点标签"},
                                    "parent": {"type": "string", "description": "父节点ID（思维导图用）"},
                                    "level": {"type": "integer", "description": "层级"},
                                    "description": {"type": "string", "description": "节点描述"},
                                },
                            },
                            "description": "节点列表",
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "label": {"type": "string", "description": "边标签"},
                                },
                            },
                            "description": "边列表（流程图/架构图用）",
                        },
                        "phases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "phase": {"type": "string", "description": "阶段名称"},
                                    "start_time": {"type": "string", "description": "开始时间"},
                                    "end_time": {"type": "string", "description": "结束时间"},
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "该阶段的事件/项",
                                    },
                                },
                            },
                            "description": "阶段列表（时间线用）",
                        },
                        "lanes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "泳道名称"},
                                    "steps": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "action": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                            "description": "泳道列表（泳道图用）",
                        },
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "description": "章节列表（视觉化总结用）",
                        },
                    },
                    "description": "图表内容数据",
                },
                "style": {
                    "type": "string",
                    "enum": ["default", "dark", "colorful", "minimal"],
                    "default": "default",
                    "description": "图表样式",
                },
            },
            "required": ["diagram_type", "title", "content"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "diagram_type": {"type": "string", "description": "图表类型"},
                "title": {"type": "string", "description": "图表标题"},
                "mermaid_code": {"type": "string", "description": "Mermaid.js 代码"},
                "svg_placeholder": {"type": "string", "description": "SVG 占位说明"},
                "description": {"type": "string", "description": "图表说明"},
                "nodes_count": {"type": "integer", "description": "节点数量"},
                "edges_count": {"type": "integer", "description": "连接数量"},
            },
            "required": ["diagram_type", "mermaid_code", "description"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["canvas", "visualization", "generation", "mermaid", "L5"],
            layer=LayerTag.L5_AGENT_RUNTIME,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=500,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=30,
        ),
    )


def _build_mermaid_flowchart(
    nodes: list[dict],
    edges: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建流程图 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    if style == "dark":
        lines.append("%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#1a1a2e','primaryTextColor':'#fff','primaryBorderColor':'#0f3460','lineColor':'#e94560','secondaryColor':'#16213e','tertiaryColor':'#0f3460'}}}%%")
    lines.append("flowchart TD")
    node_ids: set[str] = set()
    for n in nodes:
        nid = n.get("id", "")
        label = n.get("label", nid)
        desc = n.get("description", "")
        display = f"{label}<br/><small>{desc}</small>" if desc else label
        lines.append(f"    {nid}[\"{display}\"]")
        node_ids.add(nid)
    for e in edges:
        f = e.get("from", "")
        t = e.get("to", "")
        label = e.get("label", "")
        if f and t:
            if label:
                lines.append(f"    {f} -->|\"{label}\"| {t}")
            else:
                lines.append(f"    {f} --> {t}")
    return "\n".join(lines)


def _build_mermaid_mindmap(
    nodes: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建思维导图 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    if style == "colorful":
        lines.append("%%{init:{'theme':'base','themeVariables':{'primaryColor':'#FFEAA7','primaryTextColor':'#2d3436','primaryBorderColor':'#fdcb6e','lineColor':'#636e72','secondaryColor':'#dfe6e9','tertiaryColor':'#b2bec3'}}}%%")
    lines.append("mindmap")
    root = [n for n in nodes if not n.get("parent")]
    if root:
        r = root[0]
        lines.append(f"  {r.get('label', r.get('id', ''))}")
        children = [n for n in nodes if n.get("parent") == r.get("id")]
        for c in children:
            lines.append(f"    {c.get('label', c.get('id', ''))}")
            grandchildren = [n for n in nodes if n.get("parent") == c.get("id")]
            for g in grandchildren:
                lines.append(f"      {g.get('label', g.get('id', ''))}")
    else:
        for n in nodes:
            lines.append(f"  {n.get('label', n.get('id', ''))}")
    return "\n".join(lines)


def _build_mermaid_timeline(
    phases: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建时间线 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    if style == "minimal":
        lines.append("%%{init:{'theme':'base','themeVariables':{'lineColor':'#0984e3','textColor':'#2d3436'}}}%%")
    lines.append("timeline")
    lines.append("    title Timeline")
    for p in phases:
        phase_name = p.get("phase", "")
        items = p.get("items", [])
        lines.append(f"    {phase_name} : {', '.join(items)}")
    return "\n".join(lines)


def _build_mermaid_swimlane(
    lanes: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建泳道图 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    lines.append("flowchart LR")
    # 泳道分组
    for lane in lanes:
        name = lane.get("name", "")
        steps = lane.get("steps", [])
        lines.append(f"    subgraph {name.replace(' ', '_')}[\"{name}\"]")
        prev = None
        for s in steps:
            sid = s.get("label", "").replace(" ", "_")
            action = s.get("action", "")
            display = f"{sid}<br/><small>{action}</small>" if action else sid
            lines.append(f"        {sid}[\"{display}\"]")
            if prev:
                lines.append(f"    {prev} --> {sid}")
            prev = sid
        lines.append("    end")
    # 跨泳道连接
    if len(lanes) >= 2:
        for i in range(len(lanes) - 1):
            l1_steps = lanes[i].get("steps", [])
            l2_steps = lanes[i + 1].get("steps", [])
            if l1_steps and l2_steps:
                last = l1_steps[-1].get("label", "").replace(" ", "_")
                first = l2_steps[0].get("label", "").replace(" ", "_")
                lines.append(f"    {last} -.->|传递| {first}")
    return "\n".join(lines)


def _build_mermaid_architecture(
    nodes: list[dict],
    edges: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建架构图 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    if style == "colorful":
        lines.append("%%{init:{'theme':'base','themeVariables':{'primaryColor':'#74b9ff','primaryTextColor':'#2d3436','primaryBorderColor':'#0984e3','lineColor':'#636e72','secondaryColor':'#a29bfe','tertiaryColor':'#fd79a8'}}}%%")
    lines.append("graph TB")
    # 按层级分组
    levels: dict[int, list[dict]] = {}
    for n in nodes:
        lvl = n.get("level", 0)
        levels.setdefault(lvl, []).append(n)
    for lvl in sorted(levels.keys()):
        group_nodes = levels[lvl]
        if len(group_nodes) > 1:
            gname = f"L{lvl}"
            lines.append(f"    subgraph {gname}[\"层 {lvl}\"]")
            for n in group_nodes:
                nid = n.get("id", "")
                label = n.get("label", nid)
                lines.append(f"        {nid}[\"{label}\"]")
            lines.append("    end")
        else:
            n = group_nodes[0]
            nid = n.get("id", "")
            label = n.get("label", nid)
            lines.append(f"    {nid}[\"{label}\"]")
    for e in edges:
        lines.append(f"    {e.get('from', '')} --> {e.get('to', '')}")
    return "\n".join(lines)


def _build_mermaid_visual_summary(
    sections: list[dict],
    title: str,
    style: str = "default",
) -> str:
    """构建视觉化总结 Mermaid 代码."""
    lines = ["---", f"title: {title}", "---"]
    lines.append("mindmap")
    lines.append(f"  {title}")
    for s in sections:
        stitle = s.get("title", "")
        items = s.get("items", [])
        lines.append(f"    {stitle}")
        for item in items:
            lines.append(f"      {item}")
    return "\n".join(lines)


async def _canvas_generation_handler(
    diagram_type: str,
    title: str,
    content: dict,
    style: str = "default",
) -> dict[str, Any]:
    """画布生成处理函数."""
    nodes = content.get("nodes", [])
    edges = content.get("edges", [])
    phases = content.get("phases", [])
    lanes = content.get("lanes", [])
    sections = content.get("sections", [])

    mermaid_code = ""
    description = ""
    nodes_count = 0
    edges_count = 0

    if diagram_type == "flowchart":
        mermaid_code = _build_mermaid_flowchart(nodes, edges, title, style)
        nodes_count = len(nodes)
        edges_count = len(edges)
        description = f"流程图「{title}」：{nodes_count} 个节点，{edges_count} 条连接"
    elif diagram_type == "mindmap":
        mermaid_code = _build_mermaid_mindmap(nodes, title, style)
        nodes_count = len(nodes)
        description = f"思维导图「{title}」：{nodes_count} 个节点"
    elif diagram_type == "timeline":
        mermaid_code = _build_mermaid_timeline(phases, title, style)
        nodes_count = len(phases)
        description = f"时间线「{title}」：{nodes_count} 个阶段"
    elif diagram_type == "swimlane":
        mermaid_code = _build_mermaid_swimlane(lanes, title, style)
        nodes_count = sum(len(l.get("steps", [])) for l in lanes)
        edges_count = len(lanes)
        description = f"泳道图「{title}」：{len(lanes)} 个泳道，{nodes_count} 个步骤"
    elif diagram_type == "architecture":
        mermaid_code = _build_mermaid_architecture(nodes, edges, title, style)
        nodes_count = len(nodes)
        edges_count = len(edges)
        description = f"架构图「{title}」：{nodes_count} 个组件，{edges_count} 条依赖"
    elif diagram_type == "visual_summary":
        mermaid_code = _build_mermaid_visual_summary(sections, title, style)
        nodes_count = sum(len(s.get("items", [])) for s in sections)
        description = f"视觉化总结「{title}」：{len(sections)} 个章节，{nodes_count} 个要点"

    return {
        "diagram_type": diagram_type,
        "title": title,
        "mermaid_code": mermaid_code,
        "svg_placeholder": "Mermaid 代码已生成，前端使用 Mermaid 渲染器渲染",
        "description": description,
        "nodes_count": nodes_count,
        "edges_count": edges_count,
    }


# ============================================================
# chart_type -> generate.js tool name 映射表
# ============================================================

_CHART_TYPE_TO_GENERATE_TOOL: dict[str, str] = {
    "area_chart": "generate_area_chart",
    "bar_chart": "generate_bar_chart",
    "boxplot_chart": "generate_boxplot_chart",
    "column_chart": "generate_column_chart",
    "district_map": "generate_district_map",
    "dual_axes_chart": "generate_dual_axes_chart",
    "fishbone_diagram": "generate_fishbone_diagram",
    "flow_diagram": "generate_flow_diagram",
    "funnel_chart": "generate_funnel_chart",
    "histogram_chart": "generate_histogram_chart",
    "line_chart": "generate_line_chart",
    "liquid_chart": "generate_liquid_chart",
    "mind_map": "generate_mind_map",
    "network_graph": "generate_network_graph",
    "organization_chart": "generate_organization_chart",
    "path_map": "generate_path_map",
    "pie_chart": "generate_pie_chart",
    "pin_map": "generate_pin_map",
    "radar_chart": "generate_radar_chart",
    "sankey_chart": "generate_sankey_chart",
    "scatter_chart": "generate_scatter_chart",
    "treemap_chart": "generate_treemap_chart",
    "venn_chart": "generate_venn_chart",
    "violin_chart": "generate_violin_chart",
    "word_cloud_chart": "generate_word_cloud_chart",
    "spreadsheet": "generate_spreadsheet",
}


# ============================================================
# 图表生成工具: 基于 chart-visualization skill 的 26 种图表
# ============================================================

def _chart_generation_registration() -> ToolRegistration:
    """internal.chart_generation — 图表生成工具.

    基于 chart-visualization skill 的 26 种图表类型生成，
    支持折线图、柱状图、饼图、散点图、雷达图、桑基图等。
    调用 generate.js 脚本生成图表并返回图片 URL。
    """
    return ToolRegistration(
        name="chart_generation",
        description=(
            "图表生成工具：基于 chart-visualization 能力生成 26 种类型的图表，"
            "包括折线图、柱状图、饼图、散点图、雷达图、桑基图、思维导图、"
            "词云、维恩图、箱线图、漏斗图等。返回图表图片 URL。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": [
                        "area_chart",
                        "bar_chart",
                        "boxplot_chart",
                        "column_chart",
                        "district_map",
                        "dual_axes_chart",
                        "fishbone_diagram",
                        "flow_diagram",
                        "funnel_chart",
                        "histogram_chart",
                        "line_chart",
                        "liquid_chart",
                        "mind_map",
                        "network_graph",
                        "organization_chart",
                        "path_map",
                        "pie_chart",
                        "pin_map",
                        "radar_chart",
                        "sankey_chart",
                        "scatter_chart",
                        "treemap_chart",
                        "venn_chart",
                        "violin_chart",
                        "word_cloud_chart",
                        "spreadsheet",
                    ],
                    "description": "图表类型",
                },
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "图表数据，数组格式，每个元素为数据对象",
                },
                "title": {
                    "type": "string",
                    "description": "图表标题（可选）",
                },
                "theme": {
                    "type": "string",
                    "default": "default",
                    "description": "图表主题（可选，默认 default）",
                },
                "style": {
                    "type": "object",
                    "description": "图表样式配置（可选）",
                },
            },
            "required": ["chart_type", "data"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "description": "图表类型"},
                "title": {"type": "string", "description": "图表标题"},
                "image_url": {"type": "string", "description": "图表图片 URL"},
                "description": {"type": "string", "description": "图表说明"},
            },
            "required": ["chart_type", "image_url"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["chart", "visualization", "generation", "L5"],
            layer=LayerTag.L5_AGENT_RUNTIME,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=3000,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=20,
        ),
    )


async def _chart_generation_handler(
    chart_type: str,
    data: list[dict],
    title: str = "",
    theme: str = "default",
    style: dict | None = None,
) -> dict[str, Any]:
    """图表生成处理函数.

    将数据写入临时 JSON 文件，调用 generate.js 脚本生成图表，
    捕获输出 URL 并返回。若 Node.js 不可用则返回降级说明。
    """
    import shutil
    import tempfile

    # 映射 chart_type 到 generate.js 的 tool 名称
    tool_name = _CHART_TYPE_TO_GENERATE_TOOL.get(chart_type)
    if not tool_name:
        raise ValueError(f"不支持的图表类型: {chart_type}")

    # 检查 Node.js 是否可用
    node_path = shutil.which("node")
    if not node_path:
        # 降级返回：描述图表类型和数据
        data_summary = f"数据条目数: {len(data)}"
        if data and len(data) > 0:
            keys = list(data[0].keys())
            data_summary += f"，字段: {', '.join(keys)}"
        return {
            "chart_type": chart_type,
            "title": title or "",
            "image_url": "",
            "description": (
                f"图表类型: {chart_type}（{title or '未命名'}）\n"
                f"{data_summary}\n"
                f"（Node.js 环境未安装，图表生成需要安装 Node.js ≥18.0.0 后方可渲染）"
            ),
            "fallback": True,
        }

    # 构建 generate.js 的 spec 参数
    args: dict[str, Any] = {
        "data": data,
    }
    if title:
        args["title"] = title
    if theme:
        args["theme"] = theme
    if style:
        args["style"] = style

    spec = {
        "tool": tool_name,
        "args": args,
    }

    # 写入临时 JSON 文件
    tmp_dir = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, f"chart_spec_{id(spec)}.json")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)

    generate_script = (
        r"c:\Users\86187\.trae-cn\skills\chart-visualization\scripts\generate.js"
    )

    try:
        # 异步调用 generate.js 脚本
        proc = await asyncio.create_subprocess_exec(
            node_path,
            generate_script,
            tmp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"图表生成失败 (exit={proc.returncode}): {error_msg}"
            )

        image_url = stdout.decode("utf-8", errors="replace").strip()
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    return {
        "chart_type": chart_type,
        "title": title or "",
        "image_url": image_url,
        "description": f"已生成 {chart_type} 图表「{title or '未命名'}」",
    }


# ============================================================
# 共享工具 (1)
# ============================================================

def _literature_trace_registration() -> ToolRegistration:
    """internal.literature_trace — 文献溯源追踪.

    注意：L6 设计文档中的 knowledge_retrieve 在详细 Schema 中对应为 literature_trace。
    """
    return ToolRegistration(
        name="literature_trace",
        description="文献溯源追踪：追踪知识来源的文献引用链，验证知识点的文献支撑度，构建引证网络。",
        input_schema={
            "type": "object",
            "properties": {
                "kp_id": {"type": "string", "description": "知识点ID"},
                "doi": {"type": "string", "description": "文献DOI（可选，与kp_id二选一）"},
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 3,
                    "description": "溯源深度（引用链层数）",
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否包含文献元数据",
                },
            },
            "additionalProperties": False,
            "required": [],
        },
        output_schema={
            "type": "object",
            "properties": {
                "source_chain": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "depth": {"type": "integer"},
                            "doi": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "year": {"type": "integer"},
                            "citation_count": {"type": "integer"},
                            "verified": {"type": "boolean"},
                        },
                    },
                    "description": "溯源链",
                },
                "evidence_strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "证据强度(基于引用数和溯源深度)",
                },
                "gap_detected": {"type": "boolean", "description": "是否检测到溯源断链"},
                "total_sources": {"type": "integer"},
            },
            "required": ["source_chain", "evidence_strength", "total_sources"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["literature", "trace", "provenance", "CC3", "shared"],
            layer=LayerTag.CC3_PROVENANCE,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=250,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _literature_trace_handler(
    kp_id: str = "",
    doi: str = "",
    max_depth: int = 3,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """文献溯源追踪 (stub)."""
    if not kp_id and not doi:
        kp_id = "UNKNOWN"

    # 模拟溯源链
    chain: list[dict] = []
    for depth in range(max_depth):
        entry: dict[str, Any] = {
            "depth": depth,
            "doi": f"10.1000/dy3.{kp_id or doi}.d{depth}",
            "title": f"Reference source at depth {depth} for {kp_id or doi}",
            "authors": [f"Author {depth}_1", f"Author {depth}_2"],
            "year": 2020 + depth,
            "citation_count": max(0, 50 - depth * 15),
            "verified": depth < max_depth - 1,
        }
        if not include_metadata:
            entry.pop("title")
            entry.pop("authors")
            entry.pop("year")
        chain.append(entry)

    total = len(chain)
    verified_count = sum(1 for c in chain if c.get("verified"))
    evidence = verified_count / total if total > 0 else 0.0
    # 引用数加权
    avg_citations = sum(c.get("citation_count", 0) for c in chain) / total if total > 0 else 0
    evidence = min(1.0, evidence * 0.6 + (avg_citations / 50) * 0.4)

    return {
        "source_chain": chain,
        "evidence_strength": round(evidence, 4),
        "gap_detected": not chain[-1].get("verified", False) if chain else True,
        "total_sources": total,
    }


# ============================================================
# 工具注册信息列表
# ============================================================

INTERNAL_TOOL_DEFINITIONS: list[tuple[ToolRegistration, Any]] = [
    (_bkt_compute_registration(), _bkt_compute_handler),
    (_irt_evaluate_registration(), _irt_evaluate_handler),
    (_forgetfulness_scan_registration(), _forgetfulness_scan_handler),
    (_rule_engine_check_registration(), _rule_engine_check_handler),
    (_cross_validation_registration(), _cross_validation_handler),
    (_standard_value_check_registration(), _standard_value_check_handler),
    (_fact_consistency_registration(), _fact_consistency_handler),
    (_topology_analysis_registration(), _topology_analysis_handler),
    (_path_simulation_registration(), _path_simulation_handler),
    (_resource_matching_registration(), _resource_matching_handler),
    (_literature_trace_registration(), _literature_trace_handler),
    (_canvas_generation_registration(), _canvas_generation_handler),
    (_chart_generation_registration(), _chart_generation_handler),
]

# 便捷访问
INTERNAL_TOOL_NAMES = [reg.name for reg, _ in INTERNAL_TOOL_DEFINITIONS]

# 按子分类
DIAGNOSIS_TOOLS = ["bkt_compute", "irt_evaluate", "forgetfulness_scan"]
REVIEW_TOOLS = ["rule_engine_check", "cross_validation", "standard_value_check", "fact_consistency"]
GUIDANCE_TOOLS = ["topology_analysis", "path_simulation", "resource_matching"]
SHARED_TOOLS = ["literature_trace"]


def get_internal_tool(name: str) -> tuple[ToolRegistration, Any] | None:
    """按名称获取内部工具定义."""
    for reg, handler in INTERNAL_TOOL_DEFINITIONS:
        if reg.name == name:
            return reg, handler
    return None
