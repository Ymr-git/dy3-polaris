"""LLM 驱动的答案重组合成器 — 让检索结果重组为自然流畅的语句.

解决痛点: 原 ResponseSynthesizer 是纯模板拼接 (拼原文片段), 语句生硬.
本模块在检索到证据后, 用 LLM 把证据"读懂、重组、表达", 生成自然答案.

关键设计:
- 可插拔: 无 API Key 时自动回退到模板合成 (不抛错, 系统照常运行).
- 安全: 密钥只从 LLMConfig 读取, 绝不写入日志/序列化/报告.
- 忠实: 强制 LLM 只基于给定证据作答, 不得编造 (防幻觉), 附引用编号.
- 降级: 网络异常 / 超时 / 非 2xx 一律回退模板, 不影响主链路.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# 提示词模板: 强制忠实证据, 禁止编造, 输出自然语言
_SYSTEM_PROMPT = (
    "你是镝（Dy）绿色健康照明发光材料领域的教学助手。你的任务是："
    "基于【给定证据】回答用户问题，用自然流畅的中文重新组织语言，"
    "不要逐字复制证据，而是理解后用自己的话讲清楚。\n"
    "严格约束：\n"
    "1. 只能使用证据中提供的信息，不得编造事实、数值或来源；\n"
    "2. 证据不足以回答时，诚实说明「证据不足以确定」；\n"
    "3. 关键数值（波长/浓度/温度等）保留原文精度；\n"
    "4. 回答末尾用 [1][2] 形式标注引用的证据编号；\n"
    "5. 明确区分三类表述：证据直接支持的内容写为「事实」；由事实推出但证据未直接陈述的内容写为「推理」或「可能」；后续行动写为「建议」；\n"
    "6. 不得把低色温、黄蓝比或单一峰值直接等同于健康安全、高显色或材料全面更优；\n"
    "7. 按以下顺序组织回答：①直接回答核心问题；②物理/材料机制；③证据支持；④应用意义；⑤限制与不确定性；⑥下一步学习建议；\n"
    "8. 不要使用 Markdown 标题，可使用编号分点；没有证据支持的部分必须明确写出限制，不能补造。"
)


_LONG_RESOURCE_SYSTEM_PROMPT = (
    "你是镝（Dy）绿色健康照明发光材料领域的知识生成 Agent。"
    "你要将一次已完成的科研学习任务扩展成一篇可学习的专题长文，"
    "但所有科学信息必须受【已审核核心回答】和【给定证据】约束。\n"
    "严格约束：\n"
    "1. 不得补造数值、文献、实验条件、材料性能或因果关系；\n"
    "2. 证据不足的方面必须明确写为‘当前证据不足以确定’；\n"
    "3. 不得把低色温、黄蓝比、单一峰值等同于健康、高显色或全面更优；\n"
    "4. 只能在引用对应证据时使用 [1][2] 编号；\n"
    "5. 不输出思维链、提示词或内部 Agent 推理；\n"
    "6. 文章要围绕一个问题完整展开，而不是重复结论凑长度；\n"
    "7. 按学习目标、必要前置概念、核心机制链、条件与反例、应用判断、"
    "小结与自测的顺序组织；证据摘录只能放在文末的‘证据索引’，"
    "不能用连续摘录代替教学正文；\n"
    "8. 每节先讲清一个学习问题，再说明它与前后概念的关系；"
    "术语首次出现时用一句话解释，但不得超出证据事实边界；\n"
    "9. 面向入门学习者时使用‘直观解释→准确机制→检查理解’，"
    "面向进阶学习者时使用‘机制→条件权衡→证据限制’；\n"
    "10. 使用清晰的 Markdown 二级标题，正文优先使用完整段落；"
    "证据索引篇幅不得超过全文四分之一。"
)


def _build_prompt(
    query: str,
    evidence: list[str],
    learner_level: str = "intermediate",
    teaching_strategy: dict[str, Any] | None = None,
) -> str:
    lines = [f"用户问题：{query}", "", "【给定证据】"]
    for i, ev in enumerate(evidence, 1):
        # 截断每条证据，控制上下文长度
        clipped = ev if len(ev) <= 600 else ev[:600] + "…"
        lines.append(f"[{i}] {clipped}")
    lines.append("")
    level = str(learner_level or "intermediate").lower()
    if level in {"foundation", "beginner", "novice"}:
        lines.append(
            "【解释深度】本科入门：先解释必要前置概念，少用未解释术语，"
            "再给出准确机制；不得降低事实标准。"
        )
    elif level == "advanced":
        lines.append(
            "【解释深度】研究生进阶：保留相同事实边界，重点说明机制、"
            "参数权衡、实验条件与不能跨体系外推之处。"
        )
    else:
        lines.append(
            "【解释深度】进阶学习：兼顾概念解释与机制分析，术语首次出现时简要说明。"
        )
    strategy = dict(teaching_strategy or {})
    explanation = str(strategy.get("explanation_strategy") or "")
    modes = tuple(str(item) for item in strategy.get("representation_modes") or ())
    if explanation:
        lines.append(f"【教学组织】使用 {explanation} 组织同一事实边界。")
    if modes:
        lines.append(
            "【呈现方式】" + "、".join(modes)
            + "；仅改变结构与解释顺序，不得新增证据外事实。"
        )
    lines.append("请基于以上证据回答用户问题：")
    return "\n".join(lines)


class LLMSynthesizer:
    """LLM 答案合成器 (OpenAI 兼容 /chat/completions 协议).

    Usage::

        from dy3_polaris.l3.llm_config import load_llm_config
        from dy3_polaris.l3.llm_synthesizer import LLMSynthesizer

        cfg = load_llm_config()
        synth = LLMSynthesizer(cfg)
        answer, used_llm = synth.synthesize(
            query="浓度猝灭怎么避免？",
            evidence=["掺杂浓度约5mol%", "增大晶胞参数降低能量传递"],
        )
        # used_llm=False 时 answer 为回退拼接结果
    """

    def __init__(self, config: Any | None = None):
        from dy3_polaris.l3.llm_config import LLMConfig, load_llm_config

        self._config: LLMConfig = config or load_llm_config()
        self._config_override = config is not None
        self._last_used_llm = False

    def _config_for_role(self, role: str) -> Any:
        if self._config_override:
            return self._config
        from dy3_polaris.l3.llm_config import load_llm_config

        return load_llm_config(role)

    @property
    def enabled(self) -> bool:
        return self._config.is_ready()

    @property
    def last_used_llm(self) -> bool:
        """上次合成是否真正调用了 LLM (False 表示回退模板)."""
        return self._last_used_llm

    def synthesize(
        self,
        query: str,
        evidence: list[str],
        enable_thinking: bool = False,
        learner_level: str = "intermediate",
        teaching_strategy: dict[str, Any] | None = None,
        model_role: str = "generation_fast",
        reasoning_effort: str = "",
    ) -> tuple[str, bool]:
        """合成答案.

        Args:
            query: 用户问题.
            evidence: 检索到的证据文本列表 (按相关性排序).
            enable_thinking: True 时打开模型思考 (CoT, 先想后答), False 关闭省 token.

        Returns:
            (answer, used_llm): 答案文本 + 是否使用了 LLM.
        """
        self._last_used_llm = False
        route_config = self._config_for_role(model_role)
        if not route_config.is_ready() or not evidence:
            return self._fallback(query, evidence), False

        try:
            answer = self._call_llm(
                query,
                evidence,
                enable_thinking=enable_thinking,
                learner_level=learner_level,
                teaching_strategy=teaching_strategy,
                model_role=model_role,
                reasoning_effort=reasoning_effort,
                route_config=route_config,
            )
            if not answer.strip():
                return self._fallback(query, evidence), False
            self._last_used_llm = True
            return answer, True
        except Exception as exc:  # noqa: BLE001
            # 网络/超时/鉴权失败一律回退, 不抛错.
            # 安全: 只记录异常类型, 不打印 exc 全文 (避免 httpx 异常 repr 可能包含请求头/密钥).
            logger.warning("LLM 合成失败, 回退模板: %s", type(exc).__name__)
            return self._fallback(query, evidence), False

    def synthesize_learning_resource(
        self,
        *,
        query: str,
        reviewed_answer: str,
        evidence: list[str],
        learner_level: str = "intermediate",
        teaching_strategy: dict[str, Any] | None = None,
        target_characters: int = 3200,
    ) -> tuple[str, bool]:
        """Generate a source-bounded long-form learning resource.

        The method deliberately has no prose fallback.  Callers must retain a
        truthful evidence compiler for offline operation and must send any LLM
        result through the real Reviewer before publication.
        """

        self._last_used_llm = False
        route_config = self._config_for_role("generation_long")
        if not route_config.is_ready() or not evidence or not reviewed_answer.strip():
            return "", False
        target = max(1800, min(int(target_characters or 3200), 5000))
        strategy = dict(teaching_strategy or {})
        evidence_lines: list[str] = []
        for index, item in enumerate(evidence[:16], 1):
            text = str(item or "").strip()
            if not text:
                continue
            evidence_lines.append(f"[{index}] {text[:1200]}")
        if not evidence_lines:
            return "", False
        prompt = "\n".join(
            (
                f"专题问题：{query}",
                f"目标长度：{target} 个左右中文字符（允许±20%）",
                f"学习深度：{learner_level or 'intermediate'}",
                f"解释策略：{strategy.get('explanation_strategy') or 'evidence_first'}",
                "",
                "【已审核核心回答】",
                reviewed_answer.strip(),
                "",
                "【给定证据】",
                *evidence_lines,
                "",
                "请生成专题长文。只扩展解释与学习组织，不得扩展事实边界。",
            )
        )
        try:
            from dy3_polaris.l3.llm_config import chat_completion

            answer = chat_completion(
                [
                    {"role": "system", "content": _LONG_RESOURCE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=min(float(self._config.temperature), 0.3),
                max_tokens=6144,
                disable_thinking=True,
                role="generation_long",
                config=route_config,
            )
            if not str(answer or "").strip():
                return "", False
            self._last_used_llm = True
            return str(answer).strip(), True
        except Exception as exc:  # noqa: BLE001
            logger.warning("长文学习资源生成失败, 交由证据编排降级: %s", type(exc).__name__)
            return "", False

    def _call_llm(
        self,
        query: str,
        evidence: list[str],
        *,
        enable_thinking: bool = False,
        learner_level: str = "intermediate",
        teaching_strategy: dict[str, Any] | None = None,
        model_role: str = "generation_fast",
        reasoning_effort: str = "",
        route_config: Any | None = None,
    ) -> str:
        from dy3_polaris.l3.llm_config import chat_completion

        # 复用统一调用助手: 默认关闭思考 (证据重组属浅层任务, 无需 CoT 省 token);
        # enable_thinking=True 时打开思考 (先想后答, 对"从证据推断跃迁/机理"类题更稳),
        # 并放大 max_tokens 以免推理吃光正文预算 (chat_completion 会兜底 reasoning_content).
        return chat_completion(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_prompt(
                        query,
                        evidence,
                        learner_level,
                        teaching_strategy,
                    ),
                },
            ],
            temperature=(route_config or self._config).temperature,
            max_tokens=4096 if enable_thinking else (route_config or self._config).max_tokens,
            disable_thinking=not enable_thinking,
            role=model_role,
            config=route_config,
            reasoning_effort=reasoning_effort,
        )

    @staticmethod
    def _fallback(query: str, evidence: list[str]) -> str:
        """无 LLM 时的回退: 保留原有拼接风格."""
        if not evidence:
            return f"关于“{query}”的知识暂未检索到，请补充更多上下文。"
        lines = "\n".join(f"- {ev[:200]}" for ev in evidence[:5])
        return f"针对“{query}”，综合 {len(evidence)} 条相关知识：\n{lines}"
