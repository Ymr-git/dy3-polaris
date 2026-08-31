"""Dy3+ Polaris — 动态可视化生成器 (M-F8/Viz).

核心目标: **不是** 预置一张静态图, 而是根据用户任意话语 / 论文数据(描述),
实时解析并生成对应的能级图 / 电子云 / 跃迁概率可视化数据 (JSON), 交付给前端
mf8-atomic-viz 动态渲染。

动态能力体现在:
1. 意图识别: 从自然语言中识别 "画/绘制/可视化 + 能级/跃迁/电子云/轨道/概率" 等意图;
2. 术语解析: 自动把 "4f9/2"、"⁴F₉/₂"、"6h13/2"、"⁶H₁₃/₂" 等上下标/ASCII 写法
   归一化为标准能级符号, 并解析出离子、初末态、跃迁能量与波长;
3. 数据驱动: 内置常见稀土离子能级知识库, 依据解析结果裁剪/补全能级与跃迁集合,
   不同指令 → 不同图形; 支持外部结构化论文数据注入。

输出格式与前端 mf8-atomic-viz.js 完全兼容:
    {
      "viz_type": "energy" | "cloud" | "trans",
      "data": {
         "ion": "Dy³⁺", "name": "镝离子", "config": "[Xe] 4f⁹",
         "ground": "⁶H₁₅/₂", "maxE": 23000,
         "levels": [{"label","energy","j","deg","color"}],
         "transitions": [{"from","to","prob","wl","photon"}],
         "orbitals": [{"id","label","n","l","kind","rhoMax","color"}]
      },
      "note": "...",
      "parsed": {"ion", "from", "to", "viz_type", "highlight"}
    }

模块为纯标准库实现, 无外部依赖, 便于复用与测试。
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ============================================================================
# 1. 意图识别词表
# ============================================================================

#: 可视化意图触发词 (命中其一即认为含可视化意图)
_VIZ_TRIGGERS: tuple[str, ...] = (
    "画", "绘制", "绘图", "画出", "画一", "画个", "作图", "图示", "可视化",
    "能级图", "能级跃迁", "跃迁图", "跃迁示意", "光谱图", "电子云", "轨道图",
    "draw", "plot", "visualize", "diagram", "chart", "energy level",
    "transition diagram", "orbital", "electron cloud",
)

#: 可视化类型 → 关键词
_VIZ_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "energy": (
        "能级", "跃迁", "energy level", "transition", "能级图", "跃迁图",
        "能级跃迁", "光谱", "spectrum",
    ),
    "cloud": (
        "电子云", "轨道", "orbital", "electron cloud", "轨道形状", "云",
    ),
    "trans": (
        "跃迁概率", "概率", "衰变", "分支比", "transition prob", "probability",
        "branching",
    ),
}

#: 离子别名 → 标准键 (中英文 + 上下标氯等写法)
_ION_ALIASES: dict[str, str] = {
    "dy": "dy", "镝": "dy", "镝离子": "dy", "dysprosium": "dy",
    "eu": "eu", "铕": "eu", "铕离子": "eu", "europium": "eu",
    "ce": "ce", "铈": "ce", "铈离子": "ce", "cerium": "ce",
    "tb": "tb", "铽": "tb", "铽离子": "tb", "terbium": "tb",
    "er": "er", "铒": "er", "铒离子": "er", "erbium": "er",
    "yb": "yb", "镱": "yb", "镱离子": "yb", "ytterbium": "yb",
    "sm": "sm", "钐": "sm", "钐离子": "sm", "samarium": "sm",
    "nd": "nd", "钕": "nd", "钕离子": "nd", "neodymium": "nd",
}

#: 上下标 Unicode 数字 → ASCII (用于归一化 ⁴F₉/₂ 等写法)
_SUP_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}
_SUB_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
    "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}
#: 术语符号字母 (2S+1 多重度后的大写 L 字母)
_LETTERS = "SPDFGHIKLMN"

# ============================================================================
# 2. 稀土离子能级知识库 (数据驱动: 不同离子/初末态 → 不同图形)
# ============================================================================
# energy 单位 cm⁻¹; 括号内为该能级典型值 (教学取值, 量级正确即可)

_ION_KB: dict[str, dict[str, Any]] = {
    "dy": {
        "ion": "Dy³⁺", "name": "镝离子", "config": "[Xe] 4f⁹", "ground": "⁶H₁₅/₂",
        "maxE": 23000,
        "levels": [
            {"label": "⁶H₁₅/₂", "energy": 0, "j": "15/2", "deg": 16, "color": "#22c55e"},
            {"label": "⁶H₁₃/₂", "energy": 3600, "j": "13/2", "deg": 14, "color": "#84cc16"},
            {"label": "⁶H₁₁/₂", "energy": 6200, "j": "11/2", "deg": 12, "color": "#eab308"},
            {"label": "⁴I₁₅/₂", "energy": 18500, "j": "15/2", "deg": 16, "color": "#f97316"},
            {"label": "⁴F₉/₂", "energy": 21100, "j": "9/2", "deg": 10, "color": "#3b82f6"},
            {"label": "⁴F₇/₂", "energy": 22400, "j": "7/2", "deg": 8, "color": "#8b5cf6"},
        ],
        "transitions": [
            {"from": "⁴F₉/₂", "to": "⁶H₁₃/₂", "prob": 0.15, "wl": 575, "photon": "黄光"},
            {"from": "⁴F₉/₂", "to": "⁶H₁₅/₂", "prob": 0.70, "wl": 480, "photon": "蓝光"},
            {"from": "⁴F₉/₂", "to": "⁶H₁₁/₂", "prob": 0.08, "wl": 660, "photon": "红光"},
            {"from": "⁴F₇/₂", "to": "⁶H₁₅/₂", "prob": 0.42, "wl": 500, "photon": "绿光"},
            {"from": "⁴I₁₅/₂", "to": "⁶H₁₅/₂", "prob": 0.48, "wl": 480, "photon": "蓝光"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "eu": {
        "ion": "Eu³⁺", "name": "铕离子", "config": "[Xe] 4f⁶", "ground": "⁷F₀",
        "maxE": 21000,
        "levels": [
            {"label": "⁷F₀", "energy": 0, "j": "0", "deg": 1, "color": "#22c55e"},
            {"label": "⁷F₁", "energy": 360, "j": "1", "deg": 3, "color": "#84cc16"},
            {"label": "⁷F₂", "energy": 1100, "j": "2", "deg": 5, "color": "#eab308"},
            {"label": "⁷F₃", "energy": 1900, "j": "3", "deg": 7, "color": "#f97316"},
            {"label": "⁵D₀", "energy": 17200, "j": "0", "deg": 1, "color": "#dc2626"},
            {"label": "⁵D₁", "energy": 19000, "j": "1", "deg": 3, "color": "#3b82f6"},
        ],
        "transitions": [
            {"from": "⁵D₀", "to": "⁷F₂", "prob": 0.62, "wl": 612, "photon": "红光"},
            {"from": "⁵D₀", "to": "⁷F₁", "prob": 0.28, "wl": 590, "photon": "橙光"},
            {"from": "⁵D₀", "to": "⁷F₀", "prob": 0.05, "wl": 580, "photon": "橙光"},
            {"from": "⁵D₁", "to": "⁷F₁", "prob": 0.30, "wl": 535, "photon": "绿光"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "ce": {
        "ion": "Ce³⁺", "name": "铈离子", "config": "[Xe] 4f¹", "ground": "²F₅/₂",
        "maxE": 32000,
        "levels": [
            {"label": "²F₅/₂", "energy": 0, "j": "5/2", "deg": 6, "color": "#22c55e"},
            {"label": "²F₇/₂", "energy": 2200, "j": "7/2", "deg": 8, "color": "#84cc16"},
            {"label": "5d(t₂g)", "energy": 27000, "j": "", "deg": 6, "color": "#3b82f6"},
            {"label": "5d(eg)", "energy": 32000, "j": "", "deg": 4, "color": "#8b5cf6"},
        ],
        "transitions": [
            {"from": "5d(t₂g)", "to": "²F₅/₂", "prob": 0.55, "wl": 420, "photon": "蓝紫光"},
            {"from": "5d(t₂g)", "to": "²F₇/₂", "prob": 0.30, "wl": 480, "photon": "蓝光"},
            {"from": "5d(eg)", "to": "²F₅/₂", "prob": 0.35, "wl": 350, "photon": "紫外"},
        ],
        "orbitals": [{"id": "5d", "label": "5d", "n": 5, "l": 2, "kind": "m0", "rhoMax": 9, "color": "#f59e0b"}],
    },
    "tb": {
        "ion": "Tb³⁺", "name": "铽离子", "config": "[Xe] 4f⁸", "ground": "⁷F₆",
        "maxE": 21000,
        "levels": [
            {"label": "⁷F₆", "energy": 0, "j": "6", "deg": 13, "color": "#22c55e"},
            {"label": "⁷F₅", "energy": 2100, "j": "5", "deg": 11, "color": "#84cc16"},
            {"label": "⁷F₄", "energy": 3400, "j": "4", "deg": 9, "color": "#eab308"},
            {"label": "⁵D₄", "energy": 20500, "j": "4", "deg": 9, "color": "#10b981"},
        ],
        "transitions": [
            {"from": "⁵D₄", "to": "⁷F₅", "prob": 0.55, "wl": 545, "photon": "绿光"},
            {"from": "⁵D₄", "to": "⁷F₄", "prob": 0.20, "wl": 585, "photon": "黄光"},
            {"from": "⁵D₄", "to": "⁷F₆", "prob": 0.15, "wl": 490, "photon": "蓝绿光"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "er": {
        "ion": "Er³⁺", "name": "铒离子", "config": "[Xe] 4f¹¹", "ground": "⁴I₁₅/₂",
        "maxE": 20000,
        "levels": [
            {"label": "⁴I₁₅/₂", "energy": 0, "j": "15/2", "deg": 16, "color": "#22c55e"},
            {"label": "⁴I₁₃/₂", "energy": 6500, "j": "13/2", "deg": 14, "color": "#84cc16"},
            {"label": "⁴I₁₁/₂", "energy": 10200, "j": "11/2", "deg": 12, "color": "#eab308"},
            {"label": "⁴F₉/₂", "energy": 15200, "j": "9/2", "deg": 10, "color": "#f97316"},
        ],
        "transitions": [
            {"from": "⁴I₁₃/₂", "to": "⁴I₁₅/₂", "prob": 0.60, "wl": 1530, "photon": "近红外"},
            {"from": "⁴I₁₁/₂", "to": "⁴I₁₅/₂", "prob": 0.40, "wl": 980, "photon": "近红外"},
            {"from": "⁴F₉/₂", "to": "⁴I₁₅/₂", "prob": 0.50, "wl": 660, "photon": "红光"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "yb": {
        "ion": "Yb³⁺", "name": "镱离子", "config": "[Xe] 4f¹³", "ground": "²F₇/₂",
        "maxE": 11000,
        "levels": [
            {"label": "²F₇/₂", "energy": 0, "j": "7/2", "deg": 8, "color": "#22c55e"},
            {"label": "²F₅/₂", "energy": 10200, "j": "5/2", "deg": 6, "color": "#3b82f6"},
        ],
        "transitions": [
            {"from": "²F₅/₂", "to": "²F₇/₂", "prob": 1.0, "wl": 980, "photon": "近红外"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "sm": {
        "ion": "Sm³⁺", "name": "钐离子", "config": "[Xe] 4f⁵", "ground": "⁶H₅/₂",
        "maxE": 19000,
        "levels": [
            {"label": "⁶H₅/₂", "energy": 0, "j": "5/2", "deg": 6, "color": "#22c55e"},
            {"label": "⁶H₇/₂", "energy": 1000, "j": "7/2", "deg": 8, "color": "#84cc16"},
            {"label": "⁴G₅/₂", "energy": 17800, "j": "5/2", "deg": 6, "color": "#dc2626"},
        ],
        "transitions": [
            {"from": "⁴G₅/₂", "to": "⁶H₇/₂", "prob": 0.55, "wl": 604, "photon": "橙红光"},
            {"from": "⁴G₅/₂", "to": "⁶H₅/₂", "prob": 0.35, "wl": 566, "photon": "黄光"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
    "nd": {
        "ion": "Nd³⁺", "name": "钕离子", "config": "[Xe] 4f³", "ground": "⁴I₉/₂",
        "maxE": 14000,
        "levels": [
            {"label": "⁴I₉/₂", "energy": 0, "j": "9/2", "deg": 10, "color": "#22c55e"},
            {"label": "⁴I₁₁/₂", "energy": 2000, "j": "11/2", "deg": 12, "color": "#84cc16"},
            {"label": "⁴F₃/₂", "energy": 11400, "j": "3/2", "deg": 4, "color": "#3b82f6"},
        ],
        "transitions": [
            {"from": "⁴F₃/₂", "to": "⁴I₁₁/₂", "prob": 0.55, "wl": 1060, "photon": "近红外"},
            {"from": "⁴F₃/₂", "to": "⁴I₉/₂", "prob": 0.45, "wl": 880, "photon": "近红外"},
        ],
        "orbitals": [{"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"}],
    },
}

#: 通用轨道预设 (供 "电子云" 意图使用)
_GENERIC_ORBITALS: list[dict[str, Any]] = [
    {"id": "1s", "label": "1s", "n": 1, "l": 0, "kind": "m0", "rhoMax": 1, "color": "#3b82f6"},
    {"id": "2s", "label": "2s", "n": 2, "l": 0, "kind": "m0", "rhoMax": 5, "color": "#6366f1"},
    {"id": "2pz", "label": "2pz", "n": 2, "l": 1, "kind": "m0", "rhoMax": 4.2, "color": "#f59e0b"},
    {"id": "2px", "label": "2px", "n": 2, "l": 1, "kind": "cx", "rhoMax": 4.2, "color": "#ec4899"},
    {"id": "3dz2", "label": "3dz²", "n": 3, "l": 2, "kind": "m0", "rhoMax": 9, "color": "#10b981"},
    {"id": "3dxy", "label": "3dxy", "n": 3, "l": 2, "kind": "cxy", "rhoMax": 9, "color": "#8b5cf6"},
    {"id": "4fz3", "label": "4fz³", "n": 4, "l": 3, "kind": "m0", "rhoMax": 16, "color": "#06b6d4"},
]

#: 波长 (nm) → 光子颜色名称
def _photon_color(wl_nm: float) -> str:
    if wl_nm < 400:
        return "紫外"
    if wl_nm < 450:
        return "蓝紫光"
    if wl_nm < 500:
        return "蓝光"
    if wl_nm < 560:
        return "绿光"
    if wl_nm < 590:
        return "黄光"
    if wl_nm < 620:
        return "橙光"
    if wl_nm < 750:
        return "红光"
    return "近红外"


# ============================================================================
# 3. 术语解析工具
# ============================================================================

_TERM_ASCII = re.compile(r"(\d)([A-Za-z])(\d+)/(\d+)")
_TERM_UNI = re.compile(r"([0-9⁰-⁹])([A-Za-z])([0-9₀-₉]+)/([0-9₀-₉]+)")
#: 整数 J 术语符号 (无斜杠): 5D0 → ⁵D₀, 7F2 → ⁷F₂ (大写字母区分于电子组态 4f9)
_TERM_INT = re.compile(r"(\d)([A-Z])(\d+)")


def _sup_to_ascii(ch: str) -> str:
    return _SUP_MAP.get(ch, ch)


def _sub_to_ascii(ch: str) -> str:
    return _SUB_MAP.get(ch, ch)


def _label_ascii(mult: str, letter: str, jnum: str, jden: str) -> str:
    """把多重度/字母/J 转成标准上标下标能级符号, 如 4 F 9 2 → ⁴F₉/₂."""
    up = _digit_to_sup(int(mult))
    letter_u = letter.upper()
    sub = _digit_to_sub(jnum) + "/" + _digit_to_sub(jden)
    return f"{up}{letter_u}{sub}"


def _digit_to_sup(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _digit_to_sub(s: str) -> str:
    return str(s).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def normalize_term(symbol: str) -> Optional[str]:
    """把字符串归一化为标准能级符号 (⁴F₉/₂ 形式).

    支持:
      - ASCII:  "4f9/2" → "⁴F₉/₂"
      - Unicode: "⁴F₉/₂" → "⁴F₉/₂"
      - 5d(t₂g) 等晶体场/轨道标记原样返回
      - 直接已是标准符号 (含 ⁰-⁹ 上标) 原样返回

    Returns:
        归一化符号; 无法识别时返回 None。
    """
    if not symbol:
        return None
    s = str(symbol).strip()
    # 已是标准符号 (含上标 ⁰-⁹): 直接返回
    if "⁰" in s or "¹" in s or "²" in s or "³" in s or "⁴" in s or "⁵" in s or "⁶" in s or "⁷" in s or "⁸" in s or "⁹" in s:
        return s
    # 晶体场/轨道标记 (5d(t₂g)、5d(eg)、1s、2pz 等数字字母组合)
    m = re.match(r"^(\d+)([spdf])(.*)$", s)
    if m and "(" in s:
        return s
    # ASCII 术语: 4f9/2
    m = _TERM_ASCII.match(s)
    if m:
        return _label_ascii(m.group(1), m.group(2), m.group(3), m.group(4))
    # 整数 J 术语: 5D0 → ⁵D₀
    m = _TERM_INT.match(s)
    if m:
        up = _digit_to_sup(int(m.group(1)))
        return f"{up}{m.group(2).upper()}{_digit_to_sub(m.group(3))}"
    # Unicode 术语
    m = _TERM_UNI.match(s)
    if m:
        mult = _sup_to_ascii(m.group(1))
        jnum = "".join(_sub_to_ascii(c) for c in m.group(3))
        jden = "".join(_sub_to_ascii(c) for c in m.group(4))
        return _label_ascii(mult, m.group(2), jnum, jden)
    return None


def find_term_in_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """在文本中查找 "X到Y" / "X→Y" / "X→Y" 的初末态能级符号.

    Returns:
        (from_label, to_label), 均可能为 None。
    """
    # 先找分隔符: 到 / 至 / → / -> / ⟶ / ⇒
    sep_pattern = re.compile(r"(?:到|至|→|->|⟶|⇒|⇢)")
    toks: list[str] = []
    for mt in sep_pattern.split(text):
        toks.append(mt.strip())
    if len(toks) < 2:
        return None, None
    # 提取每个 token 中的术语符号
    from_lbl = _extract_term(toks[0])
    to_lbl = None
    for tk in toks[1:]:
        to_lbl = _extract_term(tk)
        if to_lbl:
            break
    return from_lbl, to_lbl


def _extract_term(token: str) -> Optional[str]:
    """从单个 token 中提取第一个术语符号并归一化."""
    # 直接尝试整个 token
    whole = normalize_term(token)
    if whole:
        return whole
    # 在 token 中查找 ASCII/Unicode 术语子串
    for pat in (_TERM_ASCII, _TERM_INT, _TERM_UNI):
        m = pat.search(token)
        if m:
            if pat is _TERM_ASCII:
                return _label_ascii(m.group(1), m.group(2), m.group(3), m.group(4))
            if pat is _TERM_INT:
                up = _digit_to_sup(int(m.group(1)))
                return f"{up}{m.group(2).upper()}{_digit_to_sub(m.group(3))}"
            mult = _sup_to_ascii(m.group(1))
            jnum = "".join(_sub_to_ascii(c) for c in m.group(3))
            jden = "".join(_sub_to_ascii(c) for c in m.group(4))
            return _label_ascii(mult, m.group(2), jnum, jden)
    return None


def detect_ion(text: str) -> Optional[str]:
    """从文本中识别离子, 返回标准键 (dy/eu/...); 未识别时返回 None."""
    low = text.lower()
    for alias, key in _ION_ALIASES.items():
        if alias.lower() in low:
            return key
    # 化学式写法: Dy3+ / Dy³⁺ / Eu3+
    m = re.search(r"\b([A-Z][a-z]?)\s*3\+\b", text)
    if m:
        sym = m.group(1).lower()
        for alias, key in _ION_ALIASES.items():
            if alias.lower() == sym:
                return key
    return None


def detect_viz_type(text: str) -> str:
    """识别可视化类型, 默认 energy."""
    low = text.lower()
    # 优先级: trans > cloud > energy (更具体优先)
    for vt in ("trans", "cloud", "energy"):
        if any(k in low for k in _VIZ_TYPE_KEYWORDS[vt]):
            return vt
    return "energy"


def _has_viz_intent(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _VIZ_TRIGGERS)


# ============================================================================
# 4. 数据生成器
# ============================================================================

def _level_map(kb_levels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {lv["label"]: lv for lv in kb_levels}


def _wavelength(e_from: float, e_to: float) -> float:
    """由初末态能级差计算发射波长 (nm)."""
    d = abs(e_to - e_from)
    if d <= 0:
        return 0.0
    return round(1e7 / d, 1)


def _build_transition(
    f_label: str,
    to_label: str,
    levels: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """为指定初末态构建一条跃迁 (依据能级差动态计算波长与光子颜色)."""
    f = levels.get(f_label)
    to = levels.get(to_label)
    if not f or not to:
        return None
    wl = _wavelength(f["energy"], to["energy"])
    photon = _photon_color(wl) if wl > 0 else "未知"
    # 相对概率: 初态激发向低能级, 默认给一个教学常用值
    prob = 0.5
    return {"from": f_label, "to": to_label, "prob": prob, "wl": wl, "photon": photon}


def _clone_kb(key: str) -> dict[str, Any]:
    """深拷贝知识库条目, 避免污染共享数据."""
    import copy

    return copy.deepcopy(_ION_KB[key])


def generate_visualization(
    query: str,
    data: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """动态生成可视化数据 (核心入口).

    Args:
        query: 用户自然语言指令 / 论文描述 (如 "画一个4f9/2到6h13/2的能级跃迁图").
        data: 可选的结构化论文数据 (直接采用, 走数据驱动而不重新解析).

    Returns:
        mf8 兼容的数据集 dict: {viz_type, data, note, parsed};
        若无法识别可视化意图返回 None。
    """
    # 结构化数据优先 (论文数据直接注入)
    if data and isinstance(data, dict):
        norm = _normalize_external_data(data)
        return {
            "viz_type": norm.get("viz_type", "energy"),
            "data": norm,
            "note": "已根据提供的结构化数据生成动态可视化",
            "parsed": {"source": "external"},
        }

    text = str(query or "").strip()
    if not text:
        return None
    if not _has_viz_intent(text):
        return None

    # 1) 离子识别
    ion_key = detect_ion(text)
    viz_type = detect_viz_type(text)
    from_lbl, to_lbl = find_term_in_text(text)

    # 2) 初末态 → 若离子已知且有对应能级, 采用该离子知识库
    if ion_key and ion_key in _ION_KB:
        kb = _clone_kb(ion_key)
    else:
        # 离子未知: 尝试从初末态推断默认用 Dy (最常见教学离子)
        kb = _clone_kb("dy")

    levels: list[dict[str, Any]] = kb["levels"]
    transitions: list[dict[str, Any]] = kb["transitions"]
    lvl_map = _level_map(levels)

    # 3) 若解析出初末态, 动态构建/凸显对应跃迁
    highlight: Optional[str] = None
    if from_lbl and to_lbl:
        # 若初末态不在知识库中, 动态补全两个能级 (保证任意指令都能出图)
        if from_lbl not in lvl_map:
            levels.insert(0, _make_fallback_level(from_lbl, 20000))
            lvl_map = _level_map(levels)
        if to_lbl not in lvl_map:
            from_e = lvl_map[from_lbl]["energy"] if from_lbl in lvl_map else 20000
            levels.append(_make_fallback_level(to_lbl, max(0, from_e - 4000)))
            lvl_map = _level_map(levels)
        tr = _build_transition(from_lbl, to_lbl, lvl_map)
        if tr:
            # 去重: 移除同初末态的旧跃迁, 插入新计算的那条
            transitions = [
                t for t in transitions
                if not (t.get("from") == from_lbl and t.get("to") == to_lbl)
            ]
            transitions.insert(0, tr)
            highlight = f"{from_lbl} → {to_lbl}"

    # 4) 组装数据
    dataset: dict[str, Any] = {
        "ion": kb["ion"],
        "name": kb["name"],
        "config": kb["config"],
        "ground": kb["ground"],
        "maxE": kb["maxE"],
        "levels": levels,
        "transitions": transitions,
        "orbitals": kb.get("orbitals") or list(_GENERIC_ORBITALS),
    }

    # 5) 说明文字 (动态生成)
    note = _compose_note(dataset, viz_type, from_lbl, to_lbl, highlight)

    return {
        "viz_type": viz_type,
        "data": dataset,
        "note": note,
        "parsed": {
            "ion": ion_key or "dy",
            "from": from_lbl,
            "to": to_lbl,
            "viz_type": viz_type,
            "highlight": highlight,
        },
    }


def _make_fallback_level(label: str, energy: int) -> dict[str, Any]:
    """为一个未在知识库中的能级符号构造占位能级."""
    return {
        "label": label,
        "energy": max(0, int(energy)),
        "j": "",
        "deg": 2,
        "color": "#8b5cf6",
    }


def _compose_note(
    dataset: dict[str, Any],
    viz_type: str,
    from_lbl: Optional[str],
    to_lbl: Optional[str],
    highlight: Optional[str],
) -> str:
    parts: list[str] = []
    if viz_type == "energy":
        parts.append(f"{dataset['ion']}（{dataset['name']}）能级跃迁图")
        if highlight:
            parts.append(f"高亮跃迁 {highlight}")
        top = max(dataset["transitions"], key=lambda t: t.get("prob", 0)) if dataset["transitions"] else None
        if top:
            parts.append(f"主跃迁 {top['from']}→{top['to']} · {top['wl']}nm · {top['photon']}")
    elif viz_type == "cloud":
        parts.append(f"{dataset['ion']} 电子云形状 (依据量子数动态渲染)")
    else:
        parts.append(f"{dataset['ion']} 跃迁概率 / 分支比")
        if highlight:
            parts.append(f"高亮 {highlight}")
    return " · ".join(parts)


def _normalize_external_data(data: dict[str, Any]) -> dict[str, Any]:
    """规范化外部结构化数据 (论文数据), 补缺省字段, 保证 mf8 可渲染."""
    out: dict[str, Any] = dict(data)
    out.setdefault("ion", "M³⁺")
    out.setdefault("name", "稀土离子")
    out.setdefault("config", "")
    out.setdefault("ground", "")
    # 防御: levels/transitions 若为非法类型 (非 list), 兜底为空列表, 避免 .get 崩溃
    raw_levels = out.get("levels") or []
    if not isinstance(raw_levels, list):
        raw_levels = []
    levels = [lv for lv in raw_levels if isinstance(lv, dict)]
    out["levels"] = [
        {
            "label": str(lv.get("label") or lv.get("symbol") or lv.get("name") or "?"),
            "energy": float(lv.get("energy", lv.get("e", 0))),
            "j": str(lv.get("j", "") or ""),
            "deg": int(lv.get("deg", lv.get("degeneracy", 2))),
            "color": str(lv.get("color", "#3b82f6")),
        }
        for lv in levels
    ]
    if levels:
        out.setdefault("maxE", max(float(lv.get("energy", lv.get("e", 0))) for lv in levels) * 1.08)
    raw_transitions = out.get("transitions") or []
    if not isinstance(raw_transitions, list):
        raw_transitions = []
    transitions = [t for t in raw_transitions if isinstance(t, dict)]
    out["transitions"] = [
        {
            "from": str(t.get("from") or ""),
            "to": str(t.get("to") or ""),
            "prob": float(t.get("prob", t.get("probability", t.get("branching", 0.5)))),
            "wl": float(t.get("wl", t.get("wavelength", _wavelength(
                _level_map(out["levels"]).get(str(t.get("from")), {}).get("energy", 0),
                _level_map(out["levels"]).get(str(t.get("to")), {}).get("energy", 0),
            )))),
            "photon": str(t.get("photon", "") or "光"),
        }
        for t in transitions
    ]
    out.setdefault("orbitals", list(_GENERIC_ORBITALS))
    return out


# ============================================================================
# 5. 面向 API 的便捷封装
# ============================================================================

def generate_for_api(
    query: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """API 封装: 返回始终为 dict, 未命中意图时给出明确结构, 便于前端兜底."""
    result = generate_visualization(query, data)
    if result is None:
        return {"hit": False, "viz_type": None, "data": None, "note": "", "parsed": {}}
    result["hit"] = True
    return result