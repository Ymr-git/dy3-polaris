"""澄清提问引擎 — 仅在系统难以理解用户请求时提问.

设计原则 (观察为主, 提问为辅):
- 系统主要通过语料观察 (extract) / 行为数据 / 画像推理理解用户, 不主动盘问.
- 主动提问仅在**对用户提问或工作难以理解/意图模糊**时触发:
  1. 意图无法识别 (fallback / off-topic 边界)
  2. 实质问题但主题不明确 (query 无可用主题词)
  3. 请求缺少必要信息 (需要澄清才能执行)
- 频率上限 (max_per_session) 防止打扰; 所有问题可跳过; 敏感话题不进题库.

触发由调用方传入 ``context["ambiguous"]`` (bool) 或
``context["clarify"]`` (dict, 含 intent/详情), 本引擎不做主动时机判断.
"""
from __future__ import annotations

from typing import Any

from dy3_polaris.l2.user_understanding.models import UnderstandingProfile

# 意图模糊时按类型给出澄清问题 (人性化设计: 先复述理解 → 引导补全, 作为对用户问题的自然补充)
_CLARIFY_BY_INTENT: dict[str, dict[str, Any]] = {
    "query": {
        "question": "你这个问题有点笼统，我想确认一下你说的方向，免得答偏：你想了解的是哪个知识点或方面？",
        "options": ["发光机理", "制备与合成", "能级与光谱", "实际应用"],
        "slot_key": "topic_clarify",
    },
    "practice": {
        "question": "想练哪块？你可以指个知识点或题型，比如「练浓度猝灭」或「出5道题」，我按你的目标来。",
        "options": ["按薄弱点出题", "按兴趣出题", "随机出题"],
        "slot_key": "practice_scope",
    },
    "recommend": {
        "question": "你说得还不够具体，方便的话告诉我：今天是想巩固基础、补薄弱点，还是探索新知识？",
        "options": ["巩固基础", "补薄弱点", "探索新知识"],
        "slot_key": "direction",
    },
    "knowledge": {
        "question": "你想查哪个知识点？举个例子我就能帮你搜得好一点，比如「量子效率」或「浓度猝灭」。",
        "options": ["量子效率", "浓度猝灭", "热猝灭", "能级与光谱"],
        "slot_key": "knowledge_topic",
    },
}

# Cold-start questions are optional and selected by the first missing slot.
# They collect priors only; no answer is interpreted as demonstrated mastery.
_INITIAL_PROFILE_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "question": "为了选择合适的解释深度，你愿意说明目前的学习或工作阶段吗？",
        "options": ["本科阶段", "研究生阶段", "科研人员", "行业从业者", "跳过"],
        "slot_key": "learning_stage",
    },
    {
        "question": "这次你最希望达成什么学习或研究目标？",
        "options": ["理解基础概念", "分析材料机理", "阅读科研证据", "解决实验问题", "跳过"],
        "slot_key": "learning_goal",
    },
    {
        "question": "你的专业背景更接近哪个方向？这只作为首次教学先验，可以跳过。",
        "options": ["材料科学", "物理/光学", "化学", "光电/照明", "跨专业", "跳过"],
        "slot_key": "professional_background",
    },
    {
        "question": "你在发光材料领域目前有哪些经历？",
        "options": ["刚开始了解", "修过相关课程", "有实验经历", "有科研经历", "有行业经历", "跳过"],
        "slot_key": "domain_experience",
    },
)


class ProactiveAsker:
    """澄清式提问引擎 (无状态; 频率计数在画像上).

    仅当调用方标记 ``context["ambiguous"]`` 或提供
    ``context["clarify"]`` 时返回澄清问题, 否则返回 None (观察为主).
    """

    def __init__(self, max_per_session: int = 3) -> None:
        self._max = max_per_session

    def next_question(self, learner_id: str, profile: UnderstandingProfile | None,
                      context: dict[str, Any]) -> dict[str, Any] | None:
        """返回当前澄清问题; 无需澄清返回 None.

        Args:
            context: 调用方上下文, 关键字段:
                - ambiguous: bool — 当前请求是否意图模糊/难以理解
                - clarify: dict — 显式澄清请求 {intent: str, ...}
                - intent: str — 已识别意图 (供 clarify 模板选择)
                - detail: str — 用户原始问题文本 (用于个性化引导)

        Returns:
            {"question": str, "trigger": str, "options": [str], "slot_key": str}
        """
        prof = profile or UnderstandingProfile(learner_id=learner_id)
        if prof.proactive_asked >= self._max:
            return None

        if context.get("initial_profile"):
            declared = dict(prof.declared_background or {})
            for template in _INITIAL_PROFILE_QUESTIONS:
                slot_key = str(template["slot_key"])
                if str(declared.get(slot_key) or "").strip():
                    continue
                prof.bump_proactive_asked()
                return {
                    "question": template["question"],
                    "trigger": "initial_profile",
                    "options": list(template["options"]),
                    "slot_key": slot_key,
                    "optional": True,
                }
            return None

        # 显式澄清请求: context.clarify
        clarify = context.get("clarify") if isinstance(context.get("clarify"), dict) else None
        if clarify:
            intent = str(clarify.get("intent") or context.get("intent") or "")
            template = _CLARIFY_BY_INTENT.get(intent) or _CLARIFY_BY_INTENT["query"]
            prof.bump_proactive_asked()
            return {
                "question": _personalize(template["question"], context.get("detail")),
                "trigger": "clarify",
                "options": list(template["options"]) + ["跳过"],
                "slot_key": template["slot_key"],
            }

        # 意图模糊标记: 由前端/上层在无法理解时传入
        if context.get("ambiguous"):
            intent = str(context.get("intent") or "")
            template = _CLARIFY_BY_INTENT.get(intent) or _CLARIFY_BY_INTENT["query"]
            prof.bump_proactive_asked()
            return {
                "question": _personalize(template["question"], context.get("detail")),
                "trigger": "ambiguous",
                "options": list(template["options"]) + ["跳过"],
                "slot_key": template["slot_key"],
            }

        # 其余情况: 观察为主, 不主动打扰
        return None


def _personalize(question: str, detail: Any) -> str:
    """把用户的问题文本自然揉进澄清引导, 让追问听起来像"接着用户的话问"而非模板."""
    detail = str(detail or "").strip()
    if not detail:
        return question
    q = detail.replace("\n", " ").strip()
    if len(q) > 24:
        q = q[:24] + "…"
    return f"你提到「{q}」——{question}"
