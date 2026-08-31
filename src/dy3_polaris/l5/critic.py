"""验证器引导的迭代自纠 critic — 对标 DeepVerifier / CoRefine / SETS.

解决类级问题: 原答案质量判定依赖「字符级 Jaccard + 假辩论」, 无真实语义判断,
导致「答非所问 / 检索不相关 / 答案空词」反复出现、每次只打补丁.
本模块把「判断答案好坏」这件事独立成一个可插拔 critic, 并驱动闭环纠错.

关键设计 (类级而非单点):
- critique_answer(question, answer, context_chunks) -> dict:
    语义评审答案质量, 输出裁决 verdict ∈ {pass / fix_relevance / fix_faithfulness
    / fix_completeness / unanswerable} + 三维评分 (relevance/faithfulness/completeness).
  · 满血版 (已配 LLM key): 用 rubric 提示词让大模型做真·语义判断 (DeepVerifier 式).
  · 回退版 (无 key): 离子一致性 + 关键词覆盖 + 空词/碎片检测 (纯启发式).
  · 安全网: 即便 LLM 判 pass, 若启发式检测到「答案离子 ∉ 证据离子」等硬伤, 强制 fix_faithfulness.
- rewrite_query(question, feedback) -> str:
    把原问题改写为更适合检索的标准查询 (step-back 抽象 / 多问拆分 / 别名归一化),
    供回路「重新检索 → 重新生成」使用. 满血版走 LLM, 回退版走规则.

纯函数、无状态、可单测; 不 import agent_workers, 避免循环依赖.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("dy3_polaris.l5.critic")

# ---- 与 agent_workers 一致的稀土离子识别 (自包含副本, 避免循环 import) ----
_ION_ASCII_RE = re.compile(
    r"(?<![A-Za-z])(Dy|Eu|Ce|Tb|Yb|Er|Nd|Sm|Pr|Ho|Tm|Gd|Lu|La)"
    r"\s*[0-9³²⁴]?[+⁺\-⁻]?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ION_CN_MAP = {
    "镝": "Dy", "铕": "Eu", "铈": "Ce", "铽": "Tb", "镱": "Yb",
    "铒": "Er", "钕": "Nd", "钐": "Sm", "镨": "Pr", "钬": "Ho", "铥": "Tm",
}
_ION_SET = {
    "Dy", "Eu", "Ce", "Tb", "Yb", "Er", "Nd", "Sm", "Pr", "Ho", "Tm", "Gd", "Lu", "La",
}

# 问题里的口语/停用词 (关键词覆盖计算时排除, 与 normalize_query 精神一致)
_QUERY_STOP = re.compile(
    r"请|帮我|一下|怎么|如何|为什么|是什么|什么是|能否|可以|吗|呢|的|了|是|在|和|与|或|及|对|把|给|从|关于|有关|介绍|讲解|系统|说说|讲|一下|区别|对比|比较|影响|作用|方法|原理|机理|多少|什么|哪些|哪种|应该|会|要|能"
)

# 空词/碎片硬伤 (启发式 faithfulness 用): 化学式/数值丢失后的特征.
# 注意: 用 ASCII 显式边界 (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) 取代 \b, 因为 Python
# \b 按 Unicode \w 判定, 中文(也是 \w)会让 \b 在中文两侧失效, 导致"在 480 nm 和
# Al O 3 附近"这类中英混排碎片漏检; 且原 (m|nm|μm|µm) 会把合法 "480 nm" 误判为碎片.
_FRAGMENT_RE = re.compile(
    r"\(\s*\)|\[\s*\]|"                                  # 空括号 (下标/上标丢失)
    r"(?<![A-Za-z0-9])\d+\s*m(?![A-Za-z0-9])|"            # 数字 + 裸 "m" (应为 nm/μm, 前缀丢失)
    r"(?<![A-Za-z])[A-Z][a-z]?\s+[A-Z][a-z]?\s+\d+(?![A-Za-z0-9])"  # 拆散化学式 "Al O 3"
)

_VERDICTS = ("pass", "fix_relevance", "fix_faithfulness", "fix_completeness", "unanswerable")

# ---------------------------------------------------------------------------
# LLM 调用 (OpenAI 兼容 /chat/completions, 复用 llm_config)
# ---------------------------------------------------------------------------
def _call_llm(messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str | None:
    """通用 LLM 调用. 返回 content 或 None (未配置 / 失败).

    复用 llm_config.chat_completion: 对 DeepSeek 思考型模型关闭思考 (省 token 且
    正文直接入 content), 并兜底 reasoning_content. 评审/改写是确定性任务, 无需 CoT.
    """
    try:
        from dy3_polaris.l3.llm_config import chat_completion
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM critic 依赖缺失: %s", type(exc).__name__)
        return None
    try:
        raw = chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=True,
            role="review",
        )
        return raw or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM critic 调用失败: %s", type(exc).__name__)
        return None


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中稳健提取首个 JSON 对象 (容忍代码围栏/前后缀)."""
    s = str(text or "").strip()
    if not s:
        return None
    # 去掉 markdown 代码围栏
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# 启发式 critic (无 key 回退 + LLM 安全网)
# ---------------------------------------------------------------------------
def _extract_ions(text: str) -> set[str]:
    s = str(text or "")
    ions: set[str] = set()
    for m in _ION_ASCII_RE.finditer(s):
        sym = m.group(1).capitalize()
        if sym in _ION_SET:
            ions.add(sym)
    for ch in s:
        if ch in _ION_CN_MAP:
            ions.add(_ION_CN_MAP[ch])
    return ions


# 离子 token 归一: "dy3+"/"dy3"/"dy" 都归到裸符号 "dy", 避免带电荷标记导致
# 相关性词袋误判 (问题写 "Dy3+" 而答案写 "Dy" 时不应视为零相关).
_ION_TOKEN_RE = re.compile(r"^(dy|eu|ce|tb|yb|er|nd|sm|pr|ho|tm|gd|lu|la)[0-9]*[+\-]?$")


def _norm_term(t: str) -> str:
    m = _ION_TOKEN_RE.match(t)
    return m.group(1) if m else t


def _content_terms(text: str) -> set[str]:
    """提取内容词 (2+ 字 CJK 串 + 字母数字串), 排除口语停用词.

    额外做离子跨表示归一: 中文离子名(镝) / 带电荷标记(Dy3+) / 裸符号(Dy)
    统一映射到裸符号, 提升相关性鲁棒性.
    """
    s = _QUERY_STOP.sub(" ", str(text or ""))
    terms: set[str] = set()
    for m in re.finditer(r"[一-鿿]{2,}|[A-Za-z][A-Za-z0-9+\-]{1,}", s):
        terms.add(_norm_term(m.group(0).lower()))
    for sym in _extract_ions(s):
        terms.add(sym.lower())
    return terms


def _heuristic(question: str, answer: str, context_chunks: list[str]) -> dict[str, Any]:
    """纯启发式三维评分 (无外部依赖, 快速, 作为回退与硬伤安全网)."""
    q_terms = _content_terms(question)
    a_terms = _content_terms(answer)
    ev_text = " ".join(str(c) for c in (context_chunks or []))
    ev_terms = _content_terms(ev_text)
    q_ions = _extract_ions(question)
    a_ions = _extract_ions(answer)
    ev_ions = _extract_ions(ev_text)

    # 1. 相关性: 问题内容词在「答案+证据」中的覆盖率
    if q_terms:
        relevance = len(q_terms & (a_terms | ev_terms)) / len(q_terms)
    else:
        relevance = 0.5
    # 离子特异性 (防"张冠李戴"): 问题点名某离子, 答案却只谈别的离子 → 相关性打折
    if q_ions:
        if a_ions and not (q_ions & a_ions):
            relevance = min(relevance, 0.55)
        elif not a_ions:
            relevance = min(relevance, 0.78)  # 答案未复述问题离子, 轻度打折
        if ev_ions and not (q_ions & ev_ions):
            relevance = min(relevance, 0.42)

    # 2. 忠实性: 答案离子必须出现在证据离子中; 且无空词/碎片硬伤
    ion_ok = (not a_ions) or a_ions.issubset(ev_ions) or not ev_ions
    fragment_ok = not bool(_FRAGMENT_RE.search(answer))
    faithfulness = 1.0 if (ion_ok and fragment_ok) else 0.35

    # 3. 完整性: 答案长度 + 是否覆盖问题核心词
    length_ok = min(1.0, len(answer) / 80.0)
    if q_terms:
        coverage = len(q_terms & a_terms) / len(q_terms)
    else:
        coverage = 0.5
    completeness = 0.5 * length_ok + 0.5 * coverage

    score = round(0.4 * relevance + 0.4 * faithfulness + 0.2 * completeness, 4)

    # 问题关注离子 X, 答案却明确只谈离子 Y (Y≠X) → 张冠李戴, 最高优先级
    wrong_ion = bool(q_ions and a_ions and not (q_ions & a_ions))
    if wrong_ion:
        verdict, reason = (
            "fix_relevance",
            f"问题关注 {'/'.join(sorted(q_ions))} 离子, 答案却只谈 {'/'.join(sorted(a_ions))} 离子, 疑似张冠李戴",
        )
    elif not a_ions.issubset(ev_ions) and a_ions and ev_ions:
        verdict, reason = "fix_faithfulness", "答案提及的稀土离子未在证据中出现, 疑似张冠李戴"
    elif not fragment_ok:
        verdict, reason = "fix_faithfulness", "答案含化学式/数值碎片(上下标丢失)"
    elif relevance < 0.35 and not ev_terms:
        verdict, reason = "unanswerable", "证据为空, 不足以回答"
    elif relevance < 0.35:
        verdict, reason = "fix_relevance", "答案与问题主题相关性不足, 检索可能偏离"
    elif completeness < 0.3:
        verdict, reason = "fix_completeness", "答案过于简略, 未覆盖问题核心要点"
    else:
        verdict, reason = "pass", "相关且基本忠实"

    return {
        "verdict": verdict,
        "relevance": round(relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "completeness": round(completeness, 4),
        "score": score,
        "reason": reason,
        "engine": "heuristic",
    }


_CRITIC_SYSTEM = (
    "你是镝（Dy）绿色健康照明发光材料领域的「答案评审员」。请客观评审下面这个回答是否真正回答了用户问题。\n"
    "三维评分 (0-10 整数)：\n"
    "1. relevance 相关性：回答是否紧扣问题主题, 证据是否与问题相关 (而非答非所问/跑题到别的离子或主题)。\n"
    "2. faithfulness 忠实性：回答是否严格基于给定证据, 无编造数值/事实, 无「张冠李戴」。\n"
    "3. completeness 完整性：回答是否覆盖问题核心要点, 是否遗漏关键信息。\n"
    "裁决 verdict 取以下之一：\n"
    "- \"pass\"：相关且基本正确, 可直接采用。\n"
    "- \"fix_relevance\"：回答跑题/证据不相关, 需改写查询重新检索。\n"
    "- \"fix_faithfulness\"：回答有编造或与证据矛盾, 需依据证据重新生成。\n"
    "- \"fix_completeness\"：回答正确但过于简略/缺关键信息, 需补充。\n"
    "- \"unanswerable\"：证据不足以回答, 应诚实说明。\n"
    "只输出一个 JSON 对象, 不要任何额外文字或 markdown：\n"
    '{"verdict":"pass","relevance":8,"faithfulness":9,"completeness":7,"reason":"一句话说明理由"}'
)


def _llm_critique(question: str, answer: str, context_chunks: list[str]) -> dict[str, Any] | None:
    """满血版 LLM rubric 评审. 失败/无 key 返回 None."""
    evidence = "\n".join(
        f"[{i}] {str(c)[:400]}" for i, c in enumerate((context_chunks or [])[:5], 1)
    ) or "(无证据)"
    user = (
        f"用户问题：{question}\n\n"
        f"候选回答：{answer[:800]}\n\n"
        f"【给定证据】\n{evidence}\n\n"
        "请按 rubric 评审并输出 JSON。"
    )
    raw = _call_llm(
        [{"role": "system", "content": _CRITIC_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=256,
    )
    if not raw:
        return None
    obj = _extract_json_obj(raw)
    if not obj:
        return None
    verdict = str(obj.get("verdict", "pass")).strip().lower()
    if verdict not in _VERDICTS:
        verdict = "pass"
    rel = float(obj.get("relevance", 5) or 5) / 10.0
    faith = float(obj.get("faithfulness", 5) or 5) / 10.0
    comp = float(obj.get("completeness", 5) or 5) / 10.0
    score = round(0.4 * rel + 0.4 * faith + 0.2 * comp, 4)
    return {
        "verdict": verdict,
        "relevance": round(rel, 4),
        "faithfulness": round(faith, 4),
        "completeness": round(comp, 4),
        "score": score,
        "reason": str(obj.get("reason", ""))[:200],
        "engine": "llm",
    }


def critique_answer(
    question: str,
    answer: str,
    context_chunks: list[str],
    *,
    deps: Any | None = None,  # 预留: 未来可注入 embedding 校验等
) -> dict[str, Any]:
    """语义评审一个 (问题, 答案, 证据) 三元组, 返回裁决 + 三维评分.

    返回字段: verdict / relevance / faithfulness / completeness / score / reason / used_llm.
    """
    answer = str(answer or "").strip()
    if not answer:
        return {
            "verdict": "unanswerable", "relevance": 0.0, "faithfulness": 0.0,
            "completeness": 0.0, "score": 0.0, "reason": "空答案", "used_llm": False, "engine": "heuristic",
        }

    h = _heuristic(question, answer, context_chunks)

    # 满血版 LLM 语义判断 (主)
    llm = _llm_critique(question, answer, context_chunks)
    if llm is not None:
        # 安全网: 启发式检测到硬伤 (离子张冠李戴/碎片) 时, 强制降级为 fix_faithfulness
        if h["verdict"] == "fix_faithfulness" and llm["verdict"] == "pass":
            llm["verdict"] = "fix_faithfulness"
            llm["score"] = min(llm["score"], 0.5)
            llm["reason"] = h["reason"]
        llm["used_llm"] = True
        return llm

    # 无 key / LLM 失败 → 启发式回退
    h["used_llm"] = False
    return h


_REWRITE_SYSTEM = (
    "你是镝（Dy）绿色健康照明发光材料检索系统的「查询改写器」。给定一个用户问题与上一轮答案的问题反馈，"
    "请把问题改写为更适合检索的标准查询，用于在知识库中重新检索。\n"
    "规则：\n"
    "1. 保留用户关心的核心实体(如 Dy³⁺、浓度猝灭、能级跃迁)，不要改变原意；\n"
    "2. 若问题含多个子问题，可拆分成一个聚焦的主问题；\n"
    "3. 若反馈是「跑题/不相关」，补充更具体的主题词或把口语表述归一为标准术语；\n"
    "4. 只输出改写后的查询文本本身，不要解释、不要引号、不要编号。"
)


def _heuristic_rewrite(question: str, feedback: str) -> str:
    """启发式改写 (无 key 回退): 别名归一化 + 去口语词 + 拆分多问取首问."""
    q = str(question or "").strip()
    if not q:
        return ""
    # 多问拆分: 取第一个 "?"/"？" 或顿号/逗号前的聚焦子问
    for sep in ("?", "？", "。", "！"):
        if sep in q:
            q = q.split(sep, 1)[0].strip()
            break
    # 别名归一化 (与 agent_workers._ENTITY_ALIASES 精神一致)
    _alias = {
        "镝": "Dy3+", "镝离子": "Dy3+", "铕": "Eu3+", "铕离子": "Eu3+",
        "铽": "Tb3+", "铈": "Ce3+", "镱": "Yb3+", "铒": "Er3+",
    }
    for k, v in _alias.items():
        if k in q:
            q = q.replace(k, v)
    # 去口语词
    q = re.sub(r"请|帮我|一下|能不能|可以不可以|请问", "", q)
    q = q.strip().strip("，,。 ")
    return q


def rewrite_query(
    question: str,
    feedback: str = "",
    *,
    deps: Any | None = None,
) -> str:
    """把原问题改写为更适合检索的标准查询 (满血版走 LLM, 回退版走规则).

    返回 "" 表示无法改写 (回路据此停止, 不做无意义重试).
    """
    question = str(question or "").strip()
    if not question:
        return ""
    feedback = str(feedback or "").strip()
    user = f"用户问题：{question}\n反馈：{feedback or '答案跑题/证据不相关，请改写以提升检索相关性'}"
    raw = _call_llm(
        [{"role": "system", "content": _REWRITE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=128,
    )
    if raw:
        rewritten = raw.strip().strip('"“”\'')
        if rewritten and rewritten != question:
            return rewritten[:200]
    # 回退: 规则改写; 若与原文相同则返回 "" (避免回路死循环)
    h = _heuristic_rewrite(question, feedback)
    if h and h != question:
        return h
    return ""
