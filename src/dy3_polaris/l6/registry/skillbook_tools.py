"""11 个 L2 Skillbook 教学技能工具的 Schema 定义与 Stub 实现.

分类 (按学习科学理论):
- S01 概念锚定 (Concept Anchoring)           — Ausubel 有意义学习理论
- S02 类比迁移 (Analogy Transfer)            — Gentner 结构映射理论
- S03 苏格拉底式提问 (Socratic Questioning)  — Vygotsky 最近发展区理论
- S04 误解诊断 (Misconception Diagnosis)     — 概念转变理论 (Posner)
- S05 渐进释放 (Gradual Release)             — Pearson & Gallagher 逐步释放责任模型
- S06 实验引导 (Experiment Guidance)         — 探究式学习 (Dewey)
- S07 文献阅读引导 (Literature Reading)      — 批判性阅读理论
- S08 跨域链接 (Cross-Domain Linking)        — 迁移学习理论
- S09 可视化解释 (Visual Explanation)        — 双重编码理论 (Paivio)
- S10 巩固练习 (Consolidation Practice)      — 间隔重复效应 (Ebbinghaus)
- S11 元认知反思 (Metacognitive Reflection)  — Flavell 元认知理论

命名规范: 教学技能使用 skill_sXX_invoke (skill_ 前缀, 与 ToolRegistration pattern 一致)
Schema 规范:
  - S01-S05: input_schema / output_schema 为扁平结构 (顶层属性即字段)
  - S06-S11: input_schema 顶层为 "input" 对象, output_schema 顶层为 "output" 对象
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)

logger = logging.getLogger(__name__)


# ============================================================
# S01 — 概念锚定 (Concept Anchoring) / Ausubel 有意义学习理论
# ============================================================

def _skill_s01_invoke_registration() -> ToolRegistration:
    """skill_s01_invoke — 概念锚定.

    基于 Ausubel 有意义学习理论，将新知识与学习者已有知识结构中的
    "锚定点"建立非人为的、实质性的联系，从而促进有意义学习。
    """
    return ToolRegistration(
        name="skill_s01_invoke",
        description=(
            "概念锚定(Concept Anchoring)：基于 Ausubel 有意义学习理论，"
            "识别学习者已有知识中的锚定点，建立新旧知识的实质性联系，"
            "并生成脚手架策略促进有意义学习。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kp_id": {
                    "type": "string",
                    "description": "目标知识点(KP)唯一标识，如 DOM-A-01",
                },
                "learner_id": {
                    "type": "string",
                    "description": "学习者唯一标识",
                },
                "prior_knowledge": {
                    "type": "string",
                    "description": "学习者已有知识描述(自由文本)",
                },
                "concept_type": {
                    "type": "string",
                    "enum": ["declarative", "procedural", "conditional", "strategic"],
                    "description": "概念类型：陈述性/程序性/条件性/策略性",
                },
            },
            "required": ["kp_id", "learner_id", "prior_knowledge", "concept_type"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "anchor_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "anchor_id": {"type": "string"},
                            "description": {"type": "string"},
                            "relation_type": {
                                "type": "string",
                                "enum": ["superordinate", "subordinate", "coordinate", "analogous"],
                            },
                            "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                    },
                    "description": "识别到的锚定点列表",
                },
                "connection_strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "新旧知识连接强度(0-1)",
                },
                "scaffolding_strategy": {
                    "type": "string",
                    "description": "推荐的脚手架策略",
                },
            },
            "required": ["anchor_points", "connection_strength", "scaffolding_strategy"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "concept_anchor", "ausubel", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=200,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=80,
        ),
    )


async def _skill_s01_invoke_handler(
    kp_id: str,
    learner_id: str,
    prior_knowledge: str,
    concept_type: str,
) -> dict[str, Any]:
    """概念锚定 (stub)."""
    anchor_points = [
        {
            "anchor_id": f"ANCHOR-{kp_id}-01",
            "description": f"基于已有知识 '{prior_knowledge[:40]}' 的锚定点",
            "relation_type": "superordinate",
            "strength": 0.82,
        },
        {
            "anchor_id": f"ANCHOR-{kp_id}-02",
            "description": f"与 {concept_type} 概念并列的锚定点",
            "relation_type": "coordinate",
            "strength": 0.65,
        },
    ]
    connection_strength = round(
        sum(a["strength"] for a in anchor_points) / len(anchor_points), 4
    )
    return {
        "anchor_points": anchor_points,
        "connection_strength": connection_strength,
        "scaffolding_strategy": (
            f"先激活锚定点 ANCHOR-{kp_id}-01，再用先行组织者桥接至 {kp_id}，"
            "最后通过对比强化实质性联系。"
        ),
    }


# ============================================================
# S02 — 类比迁移 (Analogy Transfer) / Gentner 结构映射理论
# ============================================================

def _skill_s02_invoke_registration() -> ToolRegistration:
    """skill_s02_invoke — 类比迁移."""
    return ToolRegistration(
        name="skill_s02_invoke",
        description=(
            "类比迁移(Analogy Transfer)：基于 Gentner 结构映射理论，"
            "在源概念与目标概念之间建立属性/关系映射，评估迁移难度与类比距离。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_concept": {
                    "type": "string",
                    "description": "源概念(已知/熟悉的领域)",
                },
                "target_concept": {
                    "type": "string",
                    "description": "目标概念(待学习的新领域)",
                },
                "mapping_type": {
                    "type": "string",
                    "enum": ["structural", "surface", "both"],
                    "description": "映射类型：结构映射/表面映射/混合",
                },
            },
            "required": ["source_concept", "target_concept", "mapping_type"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_attr": {"type": "string"},
                            "target_attr": {"type": "string"},
                            "relation_type": {
                                "type": "string",
                                "enum": ["attribute", "relation", "system"],
                            },
                        },
                        "required": ["source_attr", "target_attr", "relation_type"],
                    },
                    "description": "源-目标属性/关系映射列表",
                },
                "transfer_difficulty": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "迁移难度(0=易, 1=难)",
                },
                "analogical_distance": {
                    "type": "string",
                    "enum": ["near", "far", "very_far"],
                    "description": "类比距离：近迁移/远迁移/极远迁移",
                },
            },
            "required": ["mappings", "transfer_difficulty", "analogical_distance"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "analogy", "gentner", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=250,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _skill_s02_invoke_handler(
    source_concept: str,
    target_concept: str,
    mapping_type: str,
) -> dict[str, Any]:
    """类比迁移 (stub)."""
    mappings = [
        {
            "source_attr": f"{source_concept}.核心属性_A",
            "target_attr": f"{target_concept}.核心属性_A'",
            "relation_type": "attribute",
        },
        {
            "source_attr": f"{source_concept}.关系_R1",
            "target_attr": f"{target_concept}.关系_R1'",
            "relation_type": "relation",
        },
    ]
    if mapping_type in ("structural", "both"):
        mappings.append({
            "source_attr": f"{source_concept}.系统结构_S",
            "target_attr": f"{target_concept}.系统结构_S'",
            "relation_type": "system",
        })

    # 迁移难度与类比距离
    difficulty_map = {"near": 0.2, "far": 0.5, "very_far": 0.85}
    analogical_distance = "far"
    transfer_difficulty = difficulty_map[analogical_distance]

    return {
        "mappings": mappings,
        "transfer_difficulty": transfer_difficulty,
        "analogical_distance": analogical_distance,
    }


# ============================================================
# S03 — 苏格拉底式提问 (Socratic Questioning) / Vygotsky ZPD
# ============================================================

def _skill_s03_invoke_registration() -> ToolRegistration:
    """skill_s03_invoke — 苏格拉底式提问."""
    return ToolRegistration(
        name="skill_s03_invoke",
        description=(
            "苏格拉底式提问(Socratic Questioning)：基于 Vygotsky 最近发展区(ZPD)理论，"
            "生成针对学习者最近发展区的递进式提问，引导学习者自主发现洞见。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "提问主题/知识点",
                },
                "learner_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "学习者当前水平",
                },
                "question_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "生成提问数量(默认3)",
                },
                "focus_area": {
                    "type": "string",
                    "description": "提问聚焦的子领域或维度",
                },
            },
            "required": ["topic", "learner_level", "focus_area"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_text": {"type": "string"},
                            "question_type": {
                                "type": "string",
                                "enum": ["clarification", "assumption", "evidence", "implication", "meta"],
                            },
                            "target_zpd_level": {
                                "type": "string",
                                "enum": ["below_zpd", "within_zpd", "above_zpd"],
                            },
                            "expected_insight": {"type": "string"},
                        },
                        "required": ["question_text", "question_type", "target_zpd_level", "expected_insight"],
                    },
                    "description": "生成的苏格拉底式提问列表",
                },
            },
            "required": ["questions"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "socratic", "vigotsky", "zpd", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=300,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _skill_s03_invoke_handler(
    topic: str,
    learner_level: str,
    focus_area: str,
    question_count: int = 3,
) -> dict[str, Any]:
    """苏格拉底式提问 (stub)."""
    question_types = ["clarification", "assumption", "evidence", "implication", "meta"]
    zpd_levels = ["below_zpd", "within_zpd", "above_zpd"]
    # 根据 learner_level 调整起始 ZPD
    level_idx = {"beginner": 0, "intermediate": 1, "advanced": 2}.get(learner_level, 1)

    questions = []
    for i in range(question_count):
        qt = question_types[i % len(question_types)]
        zpd = zpd_levels[min(level_idx + (i // 2), 2)]
        questions.append({
            "question_text": f"关于 {topic} 的{focus_area}，你为什么认为这个假设成立？(Q{i + 1})",
            "question_type": qt,
            "target_zpd_level": zpd,
            "expected_insight": f"期望学习者通过此问理解 {topic} 中 {focus_area} 的深层逻辑。",
        })

    return {"questions": questions}


# ============================================================
# S04 — 误解诊断 (Misconception Diagnosis) / 概念转变理论 (Posner)
# ============================================================

def _skill_s04_invoke_registration() -> ToolRegistration:
    """skill_s04_invoke — 误解诊断."""
    return ToolRegistration(
        name="skill_s04_invoke",
        description=(
            "误解诊断(Misconception Diagnosis)：基于 Posner 概念转变理论，"
            "分析学习者回答中的误解类型、严重度，并生成纠正策略。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "learner_response": {
                    "type": "string",
                    "description": "学习者的作答文本",
                },
                "kp_id": {
                    "type": "string",
                    "description": "相关知识点ID",
                },
                "expected_concept": {
                    "type": "string",
                    "description": "期望的正确概念描述",
                },
            },
            "required": ["learner_response", "kp_id", "expected_concept"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "misconceptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["preconception", "factual_error", "conceptual_mix", "overgeneralization"],
                            },
                            "description": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "correction_strategy": {"type": "string"},
                        },
                        "required": ["type", "description", "severity", "correction_strategy"],
                    },
                    "description": "检测到的误解列表",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "诊断置信度",
                },
            },
            "required": ["misconceptions", "confidence"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "misconception", "conceptual_change", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=200,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _skill_s04_invoke_handler(
    learner_response: str,
    kp_id: str,
    expected_concept: str,
) -> dict[str, Any]:
    """误解诊断 (stub)."""
    misconceptions = [
        {
            "type": "preconception",
            "description": f"学习者在 {kp_id} 中表现出先验概念偏差：'{learner_response[:50]}'",
            "severity": "medium",
            "correction_strategy": f"通过认知冲突展示期望概念 '{expected_concept[:50]}'，引发概念不满并引导重构。",
        },
    ]
    return {
        "misconceptions": misconceptions,
        "confidence": 0.78,
    }


# ============================================================
# S05 — 渐进释放 (Gradual Release) / Pearson & Gallagher
# ============================================================

def _skill_s05_invoke_registration() -> ToolRegistration:
    """skill_s05_invoke — 渐进释放."""
    return ToolRegistration(
        name="skill_s05_invoke",
        description=(
            "渐进释放(Gradual Release)：基于 Pearson & Gallagher 逐步释放责任模型，"
            "根据当前教学阶段(I_do/We_do/You_do_together/You_do_alone)规划下一阶段、"
            "脚手架水平和引导步骤。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kp_id": {
                    "type": "string",
                    "description": "知识点ID",
                },
                "learner_id": {
                    "type": "string",
                    "description": "学习者ID",
                },
                "current_stage": {
                    "type": "string",
                    "enum": ["I_do", "We_do", "You_do_together", "You_do_alone"],
                    "description": "当前教学阶段",
                },
                "task_difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "任务难度",
                },
            },
            "required": ["kp_id", "learner_id", "current_stage", "task_difficulty"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "next_stage": {
                    "type": "string",
                    "enum": ["I_do", "We_do", "You_do_together", "You_do_alone"],
                    "description": "推荐的下一阶段",
                },
                "scaffolding_level": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "脚手架水平(1=全支撑, 0=无支撑)",
                },
                "guided_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "引导步骤列表",
                },
                "independence_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "独立性得分(0=完全依赖, 1=完全独立)",
                },
            },
            "required": ["next_stage", "scaffolding_level", "guided_steps", "independence_score"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "gradual_release", "scaffolding", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=180,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=80,
        ),
    )


async def _skill_s05_invoke_handler(
    kp_id: str,
    learner_id: str,
    current_stage: str,
    task_difficulty: str,
) -> dict[str, Any]:
    """渐进释放 (stub)."""
    stage_order = ["I_do", "We_do", "You_do_together", "You_do_alone"]
    stage_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    # 难度高时暂缓推进
    if task_difficulty == "hard" and stage_idx < len(stage_order) - 1:
        next_stage = stage_order[stage_idx]  # 保持当前阶段
    else:
        next_stage = stage_order[min(stage_idx + 1, len(stage_order) - 1)]

    scaffolding_map = {"I_do": 0.9, "We_do": 0.6, "You_do_together": 0.3, "You_do_alone": 0.1}
    scaffolding_level = scaffolding_map.get(next_stage, 0.5)
    independence_score = round(1.0 - scaffolding_level, 4)

    guided_steps = [
        f"Step 1: 教师示范 {kp_id} 核心操作",
        f"Step 2: 师生共同练习 {kp_id}",
        f"Step 3: 学习者独立完成 {kp_id} 变式任务",
    ]

    return {
        "next_stage": next_stage,
        "scaffolding_level": scaffolding_level,
        "guided_steps": guided_steps,
        "independence_score": independence_score,
    }


# ============================================================
# S06 — 实验引导 (Experiment Guidance) / 探究式学习 (Dewey)
# NOTE: S06-S11 使用嵌套 input/output 结构
# ============================================================

def _skill_s06_invoke_registration() -> ToolRegistration:
    """skill_s06_invoke — 实验引导."""
    return ToolRegistration(
        name="skill_s06_invoke",
        description=(
            "实验引导(Experiment Guidance)：基于 Dewey 探究式学习理论，"
            "根据知识点、实验类型、安全等级和可用材料生成实验流程、安全须知与预期结果。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "kp_id": {
                            "type": "string",
                            "description": "知识点ID",
                        },
                        "experiment_type": {
                            "type": "string",
                            "enum": ["demonstration", "hands_on", "simulation", "thought_experiment"],
                            "description": "实验类型：演示/动手/仿真/思想实验",
                        },
                        "safety_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "安全等级",
                        },
                        "available_materials": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可用实验材料列表",
                        },
                    },
                    "required": ["kp_id", "experiment_type", "safety_level", "available_materials"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "procedure": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "integer"},
                                    "action": {"type": "string"},
                                    "materials": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["step", "action"],
                            },
                            "description": "实验步骤",
                        },
                        "safety_notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "安全须知",
                        },
                        "expected_outcomes": {
                            "type": "string",
                            "description": "预期实验结果",
                        },
                        "variables": {
                            "type": "object",
                            "properties": {
                                "independent": {"type": "array", "items": {"type": "string"}},
                                "dependent": {"type": "array", "items": {"type": "string"}},
                                "controlled": {"type": "array", "items": {"type": "string"}},
                            },
                            "description": "实验变量(自变量/因变量/控制变量)",
                        },
                    },
                    "required": ["procedure", "safety_notes", "expected_outcomes", "variables"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "experiment", "inquiry", "dewey", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=350,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=40,
        ),
    )


async def _skill_s06_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """实验引导 (stub)."""
    kp_id = input.get("kp_id", "UNKNOWN")
    experiment_type = input.get("experiment_type", "hands_on")
    safety_level = input.get("safety_level", "medium")
    materials = input.get("available_materials", [])

    procedure = [
        {
            "step": 1,
            "action": f"准备实验材料：{', '.join(materials[:3]) if materials else '基础材料'}",
            "materials": materials[:3] if materials else [],
        },
        {
            "step": 2,
            "action": f"按照 {experiment_type} 方式组装实验装置，验证 {kp_id} 核心原理",
            "materials": [],
        },
        {
            "step": 3,
            "action": "记录实验数据并观察现象",
            "materials": [],
        },
    ]

    safety_notes = {
        "low": ["注意保持工作区整洁"],
        "medium": ["佩戴护目镜和手套", "注意通风", "远离明火"],
        "high": ["必须在通风橱中操作", "佩戴全套防护装备", "准备应急冲洗设备", "实验前检查安全数据表(SDS)"],
    }.get(safety_level, ["注意安全"])

    return {
        "output": {
            "procedure": procedure,
            "safety_notes": safety_notes,
            "expected_outcomes": f"预期通过本实验验证 {kp_id} 的核心规律，观察到可重复的定量现象。",
            "variables": {
                "independent": ["温度", "浓度"],
                "dependent": ["反应速率"],
                "controlled": ["压强", "催化剂用量"],
            },
        }
    }


# ============================================================
# S07 — 文献阅读引导 (Literature Reading Guidance) / 批判性阅读理论
# ============================================================

def _skill_s07_invoke_registration() -> ToolRegistration:
    """skill_s07_invoke — 文献阅读引导."""
    return ToolRegistration(
        name="skill_s07_invoke",
        description=(
            "文献阅读引导(Literature Reading Guidance)：基于批判性阅读理论，"
            "为指定文献生成核心发现摘要、方法论总结、批判性问题和阅读策略。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "paper_doi": {
                            "type": "string",
                            "description": "文献DOI",
                        },
                        "reading_level": {
                            "type": "string",
                            "enum": ["introductory", "intermediate", "expert"],
                            "description": "阅读难度层级",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "聚焦阅读的领域/维度",
                        },
                    },
                    "required": ["paper_doi", "reading_level", "focus_areas"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "key_findings": {
                            "type": "string",
                            "description": "核心发现摘要",
                        },
                        "methodology_summary": {
                            "type": "string",
                            "description": "方法论总结",
                        },
                        "critical_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "批判性问题列表",
                        },
                        "reading_strategy": {
                            "type": "string",
                            "description": "推荐阅读策略",
                        },
                    },
                    "required": ["key_findings", "methodology_summary", "critical_questions", "reading_strategy"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "literature_reading", "critical_thinking", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=280,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _skill_s07_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """文献阅读引导 (stub)."""
    paper_doi = input.get("paper_doi", "10.1000/unknown")
    reading_level = input.get("reading_level", "intermediate")
    focus_areas = input.get("focus_areas", [])

    return {
        "output": {
            "key_findings": (
                f"文献 {paper_doi} 的核心发现：在聚焦 {', '.join(focus_areas) or '综合'} 维度上，"
                "研究提出了可验证的定量结论。"
            ),
            "methodology_summary": f"采用 {reading_level} 层级可理解的实验/分析方法，控制变量充分。",
            "critical_questions": [
                "样本量是否足以支撑结论的普适性？",
                "控制变量是否遗漏了潜在的混淆因素？",
                "结论是否可被独立重复验证？",
            ],
            "reading_strategy": (
                f"建议采用 {reading_level} 阅读策略：先读摘要和结论，"
                "再按 focus_areas 定位图表，最后审查方法学局限。"
            ),
        }
    }


# ============================================================
# S08 — 跨域链接 (Cross-Domain Linking) / 迁移学习理论
# ============================================================

def _skill_s08_invoke_registration() -> ToolRegistration:
    """skill_s08_invoke — 跨域链接."""
    return ToolRegistration(
        name="skill_s08_invoke",
        description=(
            "跨域链接(Cross-Domain Linking)：基于迁移学习理论，"
            "在源域与目标域之间建立知识连接，评估迁移潜力和抽象层级。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "source_domain": {
                            "type": "string",
                            "description": "源领域",
                        },
                        "target_domain": {
                            "type": "string",
                            "description": "目标领域",
                        },
                        "kp_id": {
                            "type": "string",
                            "description": "关联知识点ID",
                        },
                        "linking_type": {
                            "type": "string",
                            "enum": ["structural", "procedural", "conceptual", "methodological"],
                            "description": "链接类型",
                        },
                    },
                    "required": ["source_domain", "target_domain", "kp_id", "linking_type"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "connections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_element": {"type": "string"},
                                    "target_element": {"type": "string"},
                                    "shared_principle": {"type": "string"},
                                },
                                "required": ["source_element", "target_element", "shared_principle"],
                            },
                            "description": "跨域连接列表",
                        },
                        "transfer_potential": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "迁移潜力(0-1)",
                        },
                        "abstraction_level": {
                            "type": "string",
                            "enum": ["concrete", "mid_level", "abstract"],
                            "description": "抽象层级",
                        },
                    },
                    "required": ["connections", "transfer_potential", "abstraction_level"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "cross_domain", "transfer", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=300,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=40,
        ),
    )


async def _skill_s08_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """跨域链接 (stub)."""
    source_domain = input.get("source_domain", "DOM-A")
    target_domain = input.get("target_domain", "DOM-B")
    kp_id = input.get("kp_id", "UNKNOWN")
    linking_type = input.get("linking_type", "conceptual")

    connections = [
        {
            "source_element": f"{source_domain}::原理_P1",
            "target_element": f"{target_domain}::原理_P1'",
            "shared_principle": f"共享 {linking_type} 原则：守恒律",
        },
        {
            "source_element": f"{source_domain}::方法_M1",
            "target_element": f"{target_domain}::方法_M1'",
            "shared_principle": f"共享方法论：建模-验证范式 (关联 {kp_id})",
        },
    ]

    abstraction_level = {"structural": "abstract", "procedural": "concrete", "conceptual": "mid_level", "methodological": "abstract"}.get(
        linking_type, "mid_level"
    )
    transfer_potential = {"concrete": 0.4, "mid_level": 0.65, "abstract": 0.85}.get(abstraction_level, 0.6)

    return {
        "output": {
            "connections": connections,
            "transfer_potential": transfer_potential,
            "abstraction_level": abstraction_level,
        }
    }


# ============================================================
# S09 — 可视化解释 (Visual Explanation) / 双重编码理论 (Paivio)
# ============================================================

def _skill_s09_invoke_registration() -> ToolRegistration:
    """skill_s09_invoke — 可视化解释."""
    return ToolRegistration(
        name="skill_s09_invoke",
        description=(
            "可视化解释(Visual Explanation)：基于 Paivio 双重编码理论，"
            "为指定概念生成可视化描述、标注点、言语解释并评估双重编码得分。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "待可视化的概念",
                        },
                        "visualization_type": {
                            "type": "string",
                            "enum": ["diagram", "animation", "interactive"],
                            "description": "可视化类型：图示/动画/交互",
                        },
                        "complexity_level": {
                            "type": "string",
                            "enum": ["simple", "moderate", "complex"],
                            "description": "复杂度层级",
                        },
                    },
                    "required": ["concept", "visualization_type", "complexity_level"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "visual_description": {
                            "type": "string",
                            "description": "可视化内容描述",
                        },
                        "annotation_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label", "description"],
                            },
                            "description": "标注点列表",
                        },
                        "verbal_explanation": {
                            "type": "string",
                            "description": "配套言语解释",
                        },
                        "dual_coding_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "双重编码得分(0-1)",
                        },
                    },
                    "required": ["visual_description", "annotation_points", "verbal_explanation", "dual_coding_score"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "visualization", "dual_coding", "paivio", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=220,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _skill_s09_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """可视化解释 (stub)."""
    concept = input.get("concept", "未知概念")
    visualization_type = input.get("visualization_type", "diagram")
    complexity_level = input.get("complexity_level", "moderate")

    annotation_points = [
        {"label": "核心结构", "description": f"{concept} 的中心结构标注"},
        {"label": "关键关系", "description": f"{concept} 中元素间的关系箭头"},
    ]
    if complexity_level == "complex":
        annotation_points.append({"label": "动态变化", "description": f"{concept} 随时间/条件的演化"})

    dual_coding_score = {"simple": 0.7, "moderate": 0.8, "complex": 0.85}.get(complexity_level, 0.75)

    return {
        "output": {
            "visual_description": (
                f"以 {visualization_type} 形式呈现 {concept}：包含中心节点、"
                f"周边关系和 {complexity_level} 层级的细节标注。"
            ),
            "annotation_points": annotation_points,
            "verbal_explanation": (
                f"言语解释：{concept} 的核心在于其结构-关系-功能的统一，"
                "可视化与文字说明形成双重编码，促进长时记忆。"
            ),
            "dual_coding_score": dual_coding_score,
        }
    }


# ============================================================
# S10 — 巩固练习 (Consolidation Practice) / 间隔重复效应 (Ebbinghaus)
# ============================================================

def _skill_s10_invoke_registration() -> ToolRegistration:
    """skill_s10_invoke — 巩固练习."""
    return ToolRegistration(
        name="skill_s10_invoke",
        description=(
            "巩固练习(Consolidation Practice)：基于 Ebbinghaus 间隔重复效应，"
            "为学习者生成练习题集、间隔重复时间表和掌握标准。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "kp_id": {
                            "type": "string",
                            "description": "知识点ID",
                        },
                        "learner_id": {
                            "type": "string",
                            "description": "学习者ID",
                        },
                        "practice_type": {
                            "type": "string",
                            "enum": ["recall", "recognition", "application", "transfer"],
                            "description": "练习类型",
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "练习难度",
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "练习题数量(默认5)",
                        },
                    },
                    "required": ["kp_id", "learner_id", "practice_type", "difficulty"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "exercises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "exercise_id": {"type": "string"},
                                    "question": {"type": "string"},
                                    "type": {"type": "string"},
                                    "difficulty": {"type": "string"},
                                },
                                "required": ["exercise_id", "question"],
                            },
                            "description": "练习题列表",
                        },
                        "spacing_schedule": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "间隔重复时间表(间隔天数序列)",
                        },
                        "mastery_criteria": {
                            "type": "object",
                            "properties": {
                                "accuracy_threshold": {"type": "number"},
                                "consecutive_correct": {"type": "integer"},
                                "retention_days": {"type": "integer"},
                            },
                            "description": "掌握标准",
                        },
                    },
                    "required": ["exercises", "spacing_schedule", "mastery_criteria"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "practice", "spaced_repetition", "ebbinghaus", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=200,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _skill_s10_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """巩固练习 (stub)."""
    kp_id = input.get("kp_id", "UNKNOWN")
    practice_type = input.get("practice_type", "recall")
    difficulty = input.get("difficulty", "medium")
    count = input.get("count", 5)

    exercises = [
        {
            "exercise_id": f"EX-{kp_id}-{i + 1:02d}",
            "question": f"关于 {kp_id} 的第 {i + 1} 题 ({practice_type}/{difficulty})",
            "type": practice_type,
            "difficulty": difficulty,
        }
        for i in range(count)
    ]

    # Ebbinghaus 间隔重复: 1天, 3天, 7天, 14天, 30天
    spacing_schedule = [1, 3, 7, 14, 30]

    return {
        "output": {
            "exercises": exercises,
            "spacing_schedule": spacing_schedule,
            "mastery_criteria": {
                "accuracy_threshold": 0.85,
                "consecutive_correct": 3,
                "retention_days": 30,
            },
        }
    }


# ============================================================
# S11 — 元认知反思 (Metacognitive Reflection) / Flavell
# ============================================================

def _skill_s11_invoke_registration() -> ToolRegistration:
    """skill_s11_invoke — 元认知反思."""
    return ToolRegistration(
        name="skill_s11_invoke",
        description=(
            "元认知反思(Metacognitive Reflection)：基于 Flavell 元认知理论，"
            "引导学习者对任务完成情况进行自我评估、策略评价和改进规划，"
            "并量化元认知意识水平。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "properties": {
                        "learner_id": {
                            "type": "string",
                            "description": "学习者ID",
                        },
                        "task_summary": {
                            "type": "string",
                            "description": "任务摘要描述",
                        },
                        "performance_data": {
                            "type": "object",
                            "description": "学习者表现数据(正确率/用时/错误类型等)",
                        },
                        "reflection_prompts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选的自定义反思提示",
                        },
                    },
                    "required": ["learner_id", "task_summary", "performance_data"],
                    "additionalProperties": False,
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "object",
                    "properties": {
                        "self_assessment": {
                            "type": "string",
                            "description": "学习者自我评估结果",
                        },
                        "strategy_evaluation": {
                            "type": "string",
                            "description": "学习策略评价",
                        },
                        "improvement_plan": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "改进计划步骤",
                        },
                        "metacognitive_awareness_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "元认知意识得分(0-1)",
                        },
                    },
                    "required": ["self_assessment", "strategy_evaluation", "improvement_plan", "metacognitive_awareness_score"],
                    "additionalProperties": False,
                },
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        annotations=Dy3ToolAnnotations(
            tags=["skill", "metacognition", "reflection", "flavell", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.SKILLBOOK,
            estimated_latency_ms=250,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _skill_s11_invoke_handler(input: dict[str, Any]) -> dict[str, Any]:
    """元认知反思 (stub)."""
    learner_id = input.get("learner_id", "UNKNOWN")
    task_summary = input.get("task_summary", "")
    performance_data = input.get("performance_data", {})
    reflection_prompts = input.get("reflection_prompts", [])

    accuracy = performance_data.get("accuracy", 0.7)

    return {
        "output": {
            "self_assessment": (
                f"学习者 {learner_id} 对任务 '{task_summary[:40]}' 的自我评估："
                f"掌握度自评 {int(accuracy * 100)}%，主要薄弱环节为概念应用。"
            ),
            "strategy_evaluation": (
                "当前策略评价：使用了机械记忆为主，缺少元认知监控；"
                "建议增加自我提问和策略调整环节。"
                + (f" 自定义提示: {'; '.join(reflection_prompts)}" if reflection_prompts else "")
            ),
            "improvement_plan": [
                "在练习前明确学习目标并预测难度",
                "练习中每 5 分钟进行一次自我检查",
                "练习后复盘错误类型并调整后续策略",
            ],
            "metacognitive_awareness_score": round(min(1.0, 0.5 + accuracy * 0.4), 4),
        }
    }


# ============================================================
# 工具注册信息列表
# ============================================================

SKILLBOOK_TOOL_DEFINITIONS: list[tuple[ToolRegistration, Any]] = [
    (_skill_s01_invoke_registration(), _skill_s01_invoke_handler),
    (_skill_s02_invoke_registration(), _skill_s02_invoke_handler),
    (_skill_s03_invoke_registration(), _skill_s03_invoke_handler),
    (_skill_s04_invoke_registration(), _skill_s04_invoke_handler),
    (_skill_s05_invoke_registration(), _skill_s05_invoke_handler),
    (_skill_s06_invoke_registration(), _skill_s06_invoke_handler),
    (_skill_s07_invoke_registration(), _skill_s07_invoke_handler),
    (_skill_s08_invoke_registration(), _skill_s08_invoke_handler),
    (_skill_s09_invoke_registration(), _skill_s09_invoke_handler),
    (_skill_s10_invoke_registration(), _skill_s10_invoke_handler),
    (_skill_s11_invoke_registration(), _skill_s11_invoke_handler),
]

# 便捷访问
SKILLBOOK_TOOL_NAMES = [reg.name for reg, _ in SKILLBOOK_TOOL_DEFINITIONS]

# 按子分类
FLAT_SCHEMA_TOOLS = ["skill_s01_invoke", "skill_s02_invoke", "skill_s03_invoke", "skill_s04_invoke", "skill_s05_invoke"]
NESTED_SCHEMA_TOOLS = [
    "skill_s06_invoke",
    "skill_s07_invoke",
    "skill_s08_invoke",
    "skill_s09_invoke",
    "skill_s10_invoke",
    "skill_s11_invoke",
]


def get_skillbook_tool(name: str) -> tuple[ToolRegistration, Any] | None:
    """按名称获取 Skillbook 教学技能工具定义."""
    for reg, handler in SKILLBOOK_TOOL_DEFINITIONS:
        if reg.name == name:
            return reg, handler
    return None
