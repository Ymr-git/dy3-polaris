"""领域推演引擎 — 基于规则推未知（对标「114+256」能力，区别于检索查已知实例）.

设计规格见 docs/deductive-reasoning.md。核心：`规则 + 已知量 → 未知结论`，
推不了就诚实返回 None（缺已知量/无规则/领域外），绝不编造数值。

三类规则：
- 数值公式（A1/A2 波长↔能量换算）：可计算、可验证
- 机理因果（B1~B4）：因果链，支持双向
- 概念关系（C1~C4）：关系推演，定性
"""
from __future__ import annotations

import re

# 物理常数：E(eV) = hc / λ(nm)，hc = 1240 eV·nm
_HC_EV_NM = 1240.0


# ============================================================
# A 类 · 数值公式推演（可计算）
# ============================================================

def _deduce_wavelength_from_energy(query: str) -> str | None:
    """A1：识别「能量差 ΔE(eV) → 发射波长 λ(nm)」，λ = 1240/E。"""
    q = str(query or "")
    if not re.search(r"(波长|发射|nm|纳米|颜色|光)", q):
        return None
    m = re.search(r"(\d+\.?\d*)\s*(?:eV|ev|电子伏|电子伏特)", q)
    if not m:
        return None
    ev = float(m.group(1))
    if ev <= 0:
        return None
    nm = _HC_EV_NM / ev
    return (
        f"由 E(eV) = 1240 / λ(nm) 推得：能量差 {ev:g} eV 对应的发射波长约为 "
        f"{nm:.0f} nm。"
    )


def _deduce_energy_from_wavelength(query: str) -> str | None:
    """A2：识别「波长 λ(nm) → 光子能量 E(eV)」，E = 1240/λ。"""
    q = str(query or "")
    if not re.search(r"(能量|eV|ev|电子伏|光子)", q):
        return None
    m = re.search(r"(\d+\.?\d*)\s*(?:nm|纳米)", q)
    if not m:
        return None
    nm = float(m.group(1))
    if nm <= 0:
        return None
    ev = _HC_EV_NM / nm
    return (
        f"由 E(eV) = 1240 / λ(nm) 推得：波长 {nm:g} nm 的光子能量约为 "
        f"{ev:.2f} eV。"
    )


# ============================================================
# B/C 类 · 因果 + 关系推演（规则表：全部关键词命中 → 结论）
# ============================================================

# 机理因果规则（AND 触发，按优先级从具体到泛化）
# 触发设计：主题词（浓度/掺杂/温度/热）+ 信号词（猝灭/临界/效率/会怎样/为什么），
# 不要求「猝灭」必须出现（用户常问「会怎样」而不说「猝灭」）。
_CONC_QUENCH = (
    "浓度猝灭因果链：掺杂浓度↑ → 离子间距↓ → 无辐射能量传递↑ → 发光效率↓"
    "（超过临界浓度骤降）。反向：浓度↓ → 效率恢复。"
)
_THERMAL_QUENCH = "热猝灭因果链：温度↑ → 声子辅助无辐射跃迁↑ → 发光效率↓（一热就变暗）。"
_UPCONV = "上转换 = 反斯托克斯发光：低能（近红外）激发 → 高能（可见/紫外）发射，即把两个小台阶的能量叠成一个大台阶。"
_F4F = "4f-4f 跃迁属宇称禁戒跃迁：谱线窄、发射波长几乎不随基质变化、荧光寿命较长（微秒至毫秒量级）。"

_CAUSAL_RULES: list[tuple[tuple[str, ...], str]] = [
    (("浓度", "猝灭"), _CONC_QUENCH),
    (("掺杂", "猝灭"), _CONC_QUENCH),
    (("浓度", "临界"), _CONC_QUENCH),
    (("掺杂", "临界"), _CONC_QUENCH),
    (("浓度", "会怎样"), _CONC_QUENCH),
    (("浓度", "为什么"), _CONC_QUENCH),
    (("浓度", "效率"), _CONC_QUENCH),
    (("热", "猝灭"), _THERMAL_QUENCH),
    (("温度", "猝灭"), _THERMAL_QUENCH),
    (("温度", "效率"), _THERMAL_QUENCH),
    (("热", "效率"), _THERMAL_QUENCH),
    (("温度", "会怎样"), _THERMAL_QUENCH),
    (("上转换",), _UPCONV),
    (("反斯托克斯",), _UPCONV),
    # 4f 特性（收紧触发，避免「4F9/2 具体跃迁」被「4f」泛匹配）
    (("4f", "谱线"), _F4F),
    (("4f", "寿命"), _F4F),
    (("4f", "禁戒"), _F4F),
    (("f-f",), _F4F),
    # 能量传递（敏化剂 → 激活剂接力）
    (("能量传递",), "能量传递像接力赛：敏化剂（如 Ce³⁺）先吸收能量，再转手递给激活剂（如 Dy³⁺），由激活剂发光。"),
    # 晶场分裂（Stark 劈裂）
    (("晶场", "分裂"), "晶场使能级发生劈裂（Stark 劈裂）：简并能级分裂成子能级，改变发射波长与谱线数目。"),
    (("晶场", "劈裂"), "晶场使能级发生劈裂（Stark 劈裂）：简并能级分裂成子能级，改变发射波长与谱线数目。"),
]

# 概念关系规则（AND 触发，按优先级从具体到泛化）
_RELATION_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("黄蓝比",),
        "黄蓝比（Y/B）决定白光色温：黄蓝比高 → 色温低（暖白）；黄蓝比低 → 色温高"
        "（冷白）。这是定性方向，具体色温 K 需光谱数据。",
    ),
    (
        ("镝", "白"),
        "Dy³⁺ 蓝光（4F9/2→6H15/2，约 480 nm）+ 黄光（4F9/2→6H13/2，约 575 nm）"
        "→ 合成为白光。",
    ),
    (
        ("dy", "白"),
        "Dy³⁺ 蓝光（4F9/2→6H15/2，约 480 nm）+ 黄光（4F9/2→6H13/2，约 575 nm）"
        "→ 合成为白光。",
    ),
    (
        ("三基色",),
        "稀土发光三基色：Eu³⁺（红）、Tb³⁺（绿）、Dy³⁺（蓝+黄→白）。",
    ),
    (
        ("色温", "蓝光"),
        "高色温白光 LED → 蓝光成分高 → 对视网膜的潜在危害更大；降低色温可减少蓝光危害。",
    ),
    (
        ("蓝光", "危害"),
        "高色温白光 LED → 蓝光成分高 → 对视网膜的潜在危害更大；降低色温可减少蓝光危害。",
    ),
    # 能级差 → 颜色（由 λ=1240/ΔE 推得的定性关系）
    (
        ("能级差", "蓝"),
        "能级差大 → 发射波长短 → 偏蓝；能级差小 → 发射波长长 → 偏红（由 λ=1240/ΔE 推得）。",
    ),
    (
        ("能级差", "红"),
        "能级差大 → 发射波长短 → 偏蓝；能级差小 → 发射波长长 → 偏红（由 λ=1240/ΔE 推得）。",
    ),
    (
        ("能级差", "波长"),
        "能级差大 → 发射波长短 → 偏蓝；能级差小 → 发射波长长 → 偏红（由 λ=1240/ΔE 推得）。",
    ),
]


def _match_rules(query: str, rules: list[tuple[tuple[str, ...], str]]) -> str | None:
    """规则表匹配：第一条「全部关键词都命中」的规则返回结论，否则 None。"""
    q = str(query or "").lower()
    for keywords, conclusion in rules:
        if all(k.lower() in q for k in keywords):
            return conclusion
    return None


# 推演意图词：问「会怎样/为什么/偏X还是Y/多少」等 → 推演（区别于「是什么/机理」检索）
_DEDUCE_INTENT = re.compile(
    r"会怎样|会怎么样|超过.*会|导致|后果|偏.*还是|高还是|低还是|多少"
    r"|为什么|为何|怎么变|有何影响|怎么发生|如何发生|为什么会|为何会"
)


def deduce(query: str) -> str | None:
    """推演查询，返回结论；无法推演返回 None（由调用方诚实说明）。"""
    q = str(query or "").strip()
    if not q:
        return None
    # 1) 数值公式推演（数值 + 求波长/能量，本身即推演意图，不需意图门）
    for fn in (_deduce_wavelength_from_energy, _deduce_energy_from_wavelength):
        try:
            out = fn(q)
        except Exception:  # noqa: BLE001
            out = None
        if out:
            return out
    # 2) 因果 / 关系推演：需「推演意图词」，避免把「浓度猝灭机理是什么」
    #    这类检索查询误当推演（会导致无证据答案被判幻觉）
    if not _DEDUCE_INTENT.search(q):
        return None
    for rules in (_CAUSAL_RULES, _RELATION_RULES):
        try:
            out = _match_rules(q, rules)
        except Exception:  # noqa: BLE001
            out = None
        if out:
            return out
    return None


__all__ = ["deduce"]
