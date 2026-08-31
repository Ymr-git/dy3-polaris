"""R-03A task understanding for the collaboration runtime.

This module classifies the user's task before the four existing agents run.  It
does not choose agents, build a task plan, or alter retrieval.  The similarly
named ``l3.intent_router.IntentResult`` remains the retrieval-routing contract;
the contract below describes the user task at the L5 runtime boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import re
from typing import Any, Callable, Mapping

logger = logging.getLogger("dy3_polaris.l5.task_understanding")


class TaskMode(str, Enum):
    """Closed task-mode vocabulary frozen by R-03A."""

    FACT_FIND = "FACT_FIND"
    EXPLAIN = "EXPLAIN"
    COMPARE = "COMPARE"
    EVALUATE = "EVALUATE"
    RESEARCH_GUIDE = "RESEARCH_GUIDE"


@dataclass(frozen=True, slots=True)
class DomainEntity:
    """A deterministic entity or measurement found in the user query."""

    text: str
    entity_type: str
    value: str | float | None = None
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Internal task-understanding result produced once per request."""

    primary_intent: str
    secondary_intents: tuple[str, ...]
    task_mode: TaskMode
    domain_entities: tuple[DomainEntity, ...]
    learner_goal: str
    evidence_need: str
    risk_level: str
    ambiguity: tuple[str, ...]
    confidence: float
    required_capabilities: tuple[str, ...]
    matched_signals: tuple[str, ...] = ()
    consistency_notes: tuple[str, ...] = ()
    semantic_source: str = "deterministic"


SemanticInterpreter = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None]


_PRIMARY_INTENT = {
    TaskMode.FACT_FIND: "locate_fact",
    TaskMode.EXPLAIN: "explain_mechanism",
    TaskMode.COMPARE: "compare_alternatives",
    TaskMode.EVALUATE: "evaluate_claim",
    TaskMode.RESEARCH_GUIDE: "guide_research",
}

_LEARNER_GOAL = {
    TaskMode.FACT_FIND: "locate_and_understand_a_specific_fact",
    TaskMode.EXPLAIN: "understand_material_or_physical_mechanism",
    TaskMode.COMPARE: "compare_materials_under_explicit_criteria",
    TaskMode.EVALUATE: "judge_a_claim_with_evidence_and_limits",
    TaskMode.RESEARCH_GUIDE: "form_a_verifiable_learning_or_research_path",
}

_CAPABILITIES = {
    TaskMode.FACT_FIND: (
        "knowledge_retrieval",
        "evidence_validation",
    ),
    TaskMode.EXPLAIN: (
        "learner_modeling",
        "knowledge_retrieval",
        "mechanism_explanation",
        "evidence_validation",
    ),
    TaskMode.COMPARE: (
        "knowledge_retrieval",
        "evidence_comparison",
        "scientific_review",
        "learning_guidance",
    ),
    TaskMode.EVALUATE: (
        "knowledge_retrieval",
        "evidence_validation",
        "scientific_review",
        "uncertainty_decision",
    ),
    TaskMode.RESEARCH_GUIDE: (
        "learner_modeling",
        "knowledge_retrieval",
        "scientific_review",
        "research_guidance",
    ),
}

_MODE_PATTERNS: dict[TaskMode, tuple[re.Pattern[str], ...]] = {
    TaskMode.FACT_FIND: (
        re.compile(r"是多少|多少|什么值|哪一?个|列出|给出|查询|查找"),
        re.compile(r"波长|峰位|能级|参数|数值|色温|显色指数|CCT|CRI", re.I),
    ),
    TaskMode.EXPLAIN: (
        re.compile(r"为什么|为何|机理|机制|原理|原因|本质|怎么回事"),
        re.compile(r"如何产生|如何发生|怎样发生|如何影响|怎样影响"),
    ),
    TaskMode.COMPARE: (
        re.compile(r"比较|对比|区别|差异|相较|相比|孰优|哪个更|哪种更|优于"),
        re.compile(r"\b(?:vs\.?|versus)\b", re.I),
    ),
    TaskMode.EVALUATE: (
        re.compile(r"是否|能否|评价|评估|判断|适合|推荐|达标|可靠"),
        re.compile(r"安全|健康|风险|毒性|危害|一定|必然|绝对"),
    ),
    TaskMode.RESEARCH_GUIDE: (
        re.compile(
            r"如何设计|怎么设计|实验方案|研究方案|研究路线|下一步研究|"
            r"如果我要研究|我想研究|准备研究|下一步|从哪里开始|重点看什么"
        ),
        re.compile(r"实验|验证|制备路线|测量|表征|测试方案|研究方法"),
    ),
}

_HARD_EVALUATION_RE = re.compile(
    r"安全|健康|风险|毒性|危害|达标|一定|必然|绝对|是否优于|是否更好"
)
_DOMAIN_RE = re.compile(
    r"Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|稀土|发光|荧光|磷光|"
    r"基质|掺杂|跃迁|能级|光谱|猝灭|量子效率|色温|显色|蓝光|照明",
    re.I,
)
_ION_RE = re.compile(
    r"(?<![A-Za-z])(Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|Lu|La)"
    r"\s*(?:3\+|2\+|4\+|³⁺|²⁺|⁴⁺|3⁺|2⁺|4⁺)?(?![A-Za-z])",
    re.I,
)
_SPECTRAL_RE = re.compile(r"(?<![A-Za-z0-9])\d+[A-Z]\d+(?:/\d+)?")
_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*"
    r"(nm|K|mol%|wt%|at%|eV|meV|cm[-‐]?1|μs|ms|lm/W|lux|%)\b",
    re.I,
)
_FORMULA_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*){2,}(?::(?:Dy|Eu|Ce|Tb|Yb|Er)"
    r"(?:3\+|³⁺|3⁺)?)?(?![A-Za-z])"
)
_METRIC_RE = re.compile(r"(?<![A-Za-z])(CCT|CRI)(?![A-Za-z])", re.I)

_SECONDARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("material_mechanism", re.compile(r"机理|机制|原理|跃迁|能级|猝灭")),
    ("material_performance", re.compile(r"效率|寿命|强度|稳定|性能|浓度|基质")),
    ("healthy_lighting", re.compile(r"健康|照明|色温|显色|蓝光|CCT|CRI", re.I)),
    ("safety_boundary", re.compile(r"安全|风险|毒性|危害|达标|一定|绝对")),
    ("research_method", re.compile(r"实验|验证|制备|测量|表征|研究")),
)


def _deduplicate_entities(entities: list[DomainEntity]) -> tuple[DomainEntity, ...]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[DomainEntity] = []
    for entity in entities:
        key = (
            entity.text.casefold(),
            entity.entity_type,
            str(entity.value),
            str(entity.unit),
        )
        if key not in seen:
            seen.add(key)
            result.append(entity)
    return tuple(result)


def _extract_entities(query: str) -> tuple[DomainEntity, ...]:
    """Reuse L3 extraction and supplement notation it does not recognise."""
    entities: list[DomainEntity] = []
    try:
        from dy3_polaris.l3.intent_router import EntityExtractor

        for entity in EntityExtractor().extract(query):
            entities.append(
                DomainEntity(
                    text=str(entity.text),
                    entity_type=str(entity.entity_type),
                    value=entity.value,
                    unit=entity.unit,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("L3 entity extraction unavailable: %s", type(exc).__name__)

    for match in _ION_RE.finditer(query):
        entities.append(
            DomainEntity(match.group(0), "ion", match.group(0), None)
        )
    for match in _FORMULA_RE.finditer(query):
        text = match.group(0)
        if text.upper() not in {"CCT", "CRI"}:
            entities.append(DomainEntity(text, "material", text, None))
    for match in _SPECTRAL_RE.finditer(query):
        entities.append(
            DomainEntity(match.group(0), "spectral_term", match.group(0), None)
        )
    for match in _NUMERIC_RE.finditer(query):
        raw_value = match.group(1)
        value: str | float = raw_value
        try:
            value = float(raw_value)
        except ValueError:
            pass
        entities.append(
            DomainEntity(match.group(0), "measurement", value, match.group(2))
        )
    for match in _METRIC_RE.finditer(query):
        metric = match.group(1).upper()
        entities.append(DomainEntity(metric, "lighting_metric", metric, None))
    return _deduplicate_entities(entities)


def _mode_signals(query: str) -> dict[TaskMode, tuple[str, ...]]:
    matches: dict[TaskMode, tuple[str, ...]] = {}
    for mode, patterns in _MODE_PATTERNS.items():
        values: list[str] = []
        for pattern in patterns:
            values.extend(match.group(0) for match in pattern.finditer(query))
        matches[mode] = tuple(dict.fromkeys(values))
    return matches


def _select_mode(query: str, signals: Mapping[TaskMode, tuple[str, ...]]) -> TaskMode:
    """Resolve deterministic signals with safety and comparison precedence."""
    if _HARD_EVALUATION_RE.search(query):
        return TaskMode.EVALUATE
    if signals[TaskMode.COMPARE]:
        return TaskMode.COMPARE
    if signals[TaskMode.RESEARCH_GUIDE]:
        return TaskMode.RESEARCH_GUIDE
    if signals[TaskMode.EXPLAIN]:
        return TaskMode.EXPLAIN
    if signals[TaskMode.EVALUATE]:
        return TaskMode.EVALUATE
    return TaskMode.FACT_FIND


def _ambiguity(
    query: str,
    mode: TaskMode,
    signals: Mapping[TaskMode, tuple[str, ...]],
    entities: tuple[DomainEntity, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    active_modes = [candidate for candidate, values in signals.items() if values]
    if len(active_modes) > 1:
        issues.append("multiple_intent_signals")
    if (
        not query.strip()
        or (len(query.strip()) <= 3 and not entities)
        or (
            len(query.strip()) <= 3
            and not active_modes
            and bool(entities)
            and all(item.entity_type == "ion" for item in entities)
        )
    ):
        issues.append("missing_subject")
    if mode is TaskMode.COMPARE:
        separators = len(re.findall(r"和|与|及|vs\.?|versus|、", query, re.I))
        material_entities = [
            item for item in entities if item.entity_type in {"material", "ion"}
        ]
        if separators == 0 and len(material_entities) < 2:
            issues.append("missing_comparison_target")
    if mode is TaskMode.EVALUATE and not (
        re.search(
            r"条件|浓度|温度|波长|基质|标准|指标|CCT|CRI|色温|显色|蓝光",
            query,
            re.I,
        )
        or _NUMERIC_RE.search(query)
    ):
        issues.append("evaluation_conditions_unspecified")
    if re.search(r"健康|安全", query) and not re.search(
        r"蓝光|昼夜节律|节律|眩光|显色|色温|CCT|照度|标准|风险", query, re.I
    ):
        issues.append("health_criterion_unspecified")
    return tuple(dict.fromkeys(issues))


def _evidence_need(query: str, mode: TaskMode) -> str:
    if mode in {TaskMode.COMPARE, TaskMode.EVALUATE, TaskMode.RESEARCH_GUIDE}:
        return "high"
    if _NUMERIC_RE.search(query) or re.search(r"标准|文献|报道|数据", query):
        return "high"
    return "medium" if mode is TaskMode.EXPLAIN else "low"


def _risk_level(query: str, mode: TaskMode) -> str:
    if re.search(r"安全|健康|风险|毒性|危害|达标|标准|一定|必然|绝对", query):
        return "high"
    if mode in {TaskMode.COMPARE, TaskMode.EVALUATE, TaskMode.RESEARCH_GUIDE}:
        return "medium"
    if _NUMERIC_RE.search(query):
        return "medium"
    return "low"


def _secondary_intents(query: str, mode: TaskMode) -> tuple[str, ...]:
    result = [name for name, pattern in _SECONDARY_PATTERNS if pattern.search(query)]
    if mode is TaskMode.COMPARE:
        result.append("comparison")
    if mode is TaskMode.EVALUATE:
        result.append("claim_evaluation")
    return tuple(dict.fromkeys(result))


def _deterministic_confidence(
    query: str,
    mode: TaskMode,
    signals: Mapping[TaskMode, tuple[str, ...]],
    entities: tuple[DomainEntity, ...],
    ambiguity: tuple[str, ...],
) -> float:
    confidence = 0.48
    if signals[mode]:
        confidence += 0.18
    if entities:
        confidence += 0.08
    if _DOMAIN_RE.search(query):
        confidence += 0.08
    if len(query.strip()) >= 8:
        confidence += 0.05
    active_modes = sum(bool(values) for values in signals.values())
    if active_modes > 1:
        confidence -= min(0.12, 0.04 * (active_modes - 1))
    confidence -= min(0.15, 0.04 * len(ambiguity))
    return round(max(0.2, min(0.95, confidence)), 4)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip())
    if not cleaned:
        return None
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _default_semantic_interpreter(
    query: str,
    deterministic: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Use the configured LLM only as a bounded semantic second opinion."""
    try:
        from dy3_polaris.l3.llm_config import chat_completion, load_llm_config

        if not load_llm_config().is_ready():
            return None
        raw = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify one learning/research question. Return JSON only. "
                        "task_mode must be one of FACT_FIND, EXPLAIN, COMPARE, "
                        "EVALUATE, RESEARCH_GUIDE. Do not invent scientific facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n"
                        f"Deterministic reading: {json.dumps(dict(deterministic), ensure_ascii=False)}\n"
                        "Return task_mode, confidence, secondary_intents, ambiguity."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=220,
            disable_thinking=True,
        )
        return _extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("semantic task interpretation unavailable: %s", type(exc).__name__)
        return None


def understand_task(
    query: str,
    *,
    learner_context: Mapping[str, Any] | None = None,
    semantic_interpreter: SemanticInterpreter | None = None,
    use_llm: bool = True,
) -> IntentResult:
    """Resolve deterministic signals, optional LLM semantics, then consistency.

    The result is descriptive only.  It does not schedule agents or mutate the
    current task state, response, retrieval, review, or guidance decision.
    """
    del learner_context  # Frozen field boundary; diagnosis remains the profile authority.
    text = str(query or "").strip()
    entities = _extract_entities(text)
    signals = _mode_signals(text)
    deterministic_mode = _select_mode(text, signals)
    ambiguity = list(_ambiguity(text, deterministic_mode, signals, entities))
    confidence = _deterministic_confidence(
        text, deterministic_mode, signals, entities, tuple(ambiguity)
    )
    mode = deterministic_mode
    source = "deterministic"
    notes: list[str] = []

    signal_summary = {
        candidate.value: list(values)
        for candidate, values in signals.items()
        if values
    }
    semantic: Mapping[str, Any] | None = None
    should_interpret = bool(ambiguity) or len(signal_summary) != 1 or confidence < 0.75
    interpreter = semantic_interpreter
    if interpreter is None and use_llm and should_interpret:
        interpreter = _default_semantic_interpreter
    if interpreter is not None:
        try:
            semantic = interpreter(
                text,
                {
                    "task_mode": deterministic_mode.value,
                    "signals": signal_summary,
                    "ambiguity": list(ambiguity),
                    "confidence": confidence,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("semantic interpreter failed: %s", type(exc).__name__)
            semantic = None

    if semantic:
        raw_mode = str(semantic.get("task_mode") or "").strip().upper()
        try:
            semantic_mode = TaskMode(raw_mode)
        except ValueError:
            semantic_mode = None
            notes.append("semantic_mode_invalid")
        semantic_confidence = semantic.get("confidence", 0.0)
        try:
            semantic_confidence = max(0.0, min(1.0, float(semantic_confidence)))
        except (TypeError, ValueError):
            semantic_confidence = 0.0
        strong_rule = bool(
            _HARD_EVALUATION_RE.search(text)
            or signals[TaskMode.COMPARE]
            or signals[TaskMode.RESEARCH_GUIDE]
            or signals[TaskMode.EXPLAIN]
        )
        if semantic_mode is deterministic_mode:
            confidence = round(min(0.97, max(confidence, semantic_confidence) + 0.03), 4)
            source = "deterministic+llm"
        elif semantic_mode is not None and strong_rule:
            confidence = round(max(0.2, confidence - 0.18), 4)
            ambiguity.append("semantic_disagreement")
            notes.append(
                f"kept_{deterministic_mode.value}_over_{semantic_mode.value}"
            )
            source = "deterministic+llm_disagreement"
        elif semantic_mode is not None:
            mode = semantic_mode
            confidence = round(max(0.2, min(0.9, semantic_confidence * 0.8)), 4)
            notes.append(
                f"semantic_resolution_{deterministic_mode.value}_to_{semantic_mode.value}"
            )
            source = "llm_resolved"

    flattened_signals = tuple(
        f"{candidate.value}:{value}"
        for candidate, values in signals.items()
        for value in values
    )
    return IntentResult(
        primary_intent=_PRIMARY_INTENT[mode],
        secondary_intents=_secondary_intents(text, mode),
        task_mode=mode,
        domain_entities=entities,
        learner_goal=_LEARNER_GOAL[mode],
        evidence_need=_evidence_need(text, mode),
        risk_level=_risk_level(text, mode),
        ambiguity=tuple(dict.fromkeys(ambiguity)),
        confidence=confidence,
        required_capabilities=_CAPABILITIES[mode],
        matched_signals=flattened_signals,
        consistency_notes=tuple(notes),
        semantic_source=source,
    )


__all__ = ["DomainEntity", "IntentResult", "TaskMode", "understand_task"]
