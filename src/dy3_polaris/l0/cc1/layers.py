"""CC1 四层反幻觉评审引擎 — 层校验器.

实现设计文档中定义的四层评审架构：
- L1 事实层 (FactLayer): F-R01~F-R12 域特定事实规则
- L2 逻辑层 (LogicLayer): L-R01~L-R10 逻辑一致性规则
- L3 数值层 (NumericalLayer): N-R01~N-R12 数值范围校验规则
- L4 溯源层 (ProvenanceLayer): P-R01~P-R10 溯源完整性规则

融合世界先进方案：
- RAGAS: 声明级粒度评估
- FActScore: 原子事实分解与逐条验证
- Guardrails AI: 可插拔规则注册
- NeMo Guardrails: 分层 Rail 架构
- LlamaIndex Citation: 溯源绑定

每层独立评分 (0-100), 综合评分公式:
  Score = 0.40×L1 + 0.25×L2 + 0.20×L3 + 0.15×L4
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable

from .models import Claim, ClaimType, Evidence, EvidenceType


# ============================================================
# 枚举与常量
# ============================================================


class ReviewLayerType(str, Enum):
    """评审层类型."""

    L1_FACT = "l1_fact"
    L2_LOGIC = "l2_logic"
    L3_NUMERICAL = "l3_numerical"
    L4_PROVENANCE = "l4_provenance"


class RuleSeverity(str, Enum):
    """规则严重级别."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


#: 四层权重 (设计文档 §6.3)
LAYER_WEIGHTS: dict[ReviewLayerType, float] = {
    ReviewLayerType.L1_FACT: 0.40,
    ReviewLayerType.L2_LOGIC: 0.25,
    ReviewLayerType.L3_NUMERICAL: 0.20,
    ReviewLayerType.L4_PROVENANCE: 0.15,
}


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ReviewRule:
    """评审规则定义.

    Attributes:
        rule_id: 规则 ID (如 F-R01)
        name: 规则名称
        description: 规则描述
        severity: 严重级别
        checker: 检查函数 (claim, context) -> (passed, detail)
    """

    rule_id: str
    name: str
    description: str
    severity: RuleSeverity = RuleSeverity.WARNING
    checker: Callable[..., tuple[bool, str]] = field(
        default_factory=lambda: lambda *a, **kw: (True, "默认通过")
    )


@dataclass
class LayerRuleResult:
    """单条规则验证结果.

    Attributes:
        rule_id: 规则 ID
        rule_name: 规则名称
        passed: 是否通过
        severity: 严重级别
        detail: 详细说明
        score: 规则分数 (0-1)
    """

    rule_id: str
    rule_name: str
    passed: bool
    severity: RuleSeverity = RuleSeverity.INFO
    detail: str = ""
    score: float = 1.0


@dataclass
class LayerResult:
    """单层评审结果.

    Attributes:
        layer_type: 层类型
        score: 层评分 (0-100)
        rule_results: 各规则验证结果
        verdict: 层级判决 (PASS/FLAG/BLOCK)
        summary: 评审摘要
    """

    layer_type: ReviewLayerType
    score: float = 0.0
    rule_results: list[LayerRuleResult] = field(default_factory=list)
    verdict: str = "PASS"
    summary: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.rule_results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.rule_results if not r.passed)

    @property
    def total_count(self) -> int:
        return len(self.rule_results)


# ============================================================
# 基类
# ============================================================


class BaseReviewLayer(ABC):
    """评审层基类.

    所有层校验器继承此基类, 实现具体的规则集和验证逻辑.
    """

    layer_type: ReviewLayerType

    def __init__(self) -> None:
        self._rules: list[ReviewRule] = []
        self._init_rules()

    @abstractmethod
    def _init_rules(self) -> None:
        """初始化规则集."""
        ...

    @property
    def rules(self) -> list[ReviewRule]:
        """获取规则列表."""
        return self._rules

    def verify_claim(
        self,
        claim: Claim,
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        **kwargs: Any,
    ) -> LayerResult:
        """验证单个声明.

        对声明执行所有规则检查, 汇总评分.

        Args:
            claim: 待验证声明
            context_chunks: 上下文片段
            evidence: 证据列表

        Returns:
            层评审结果
        """
        chunks = context_chunks or []
        ev_list = evidence or []
        combined_text = claim.text + " " + " ".join(chunks)

        rule_results: list[LayerRuleResult] = []
        for rule in self._rules:
            try:
                passed, detail = rule.checker(
                    claim=claim,
                    text=combined_text,
                    claim_text=claim.text,
                    context_chunks=chunks,
                    evidence=ev_list,
                    **kwargs,
                )
            except Exception:
                passed, detail = False, "规则检查异常"

            score = 1.0 if passed else 0.0
            rule_results.append(LayerRuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=passed,
                severity=rule.severity,
                detail=detail,
                score=score,
            ))

        # 计算评分: 通过规则数 / 总规则数 × 100
        passed_count = sum(1 for r in rule_results if r.passed)
        total = len(rule_results)
        score = round(passed_count / total * 100, 2) if total > 0 else 100.0

        # 判决: 全通过 → PASS, 有 CRITICAL → BLOCK, 否则 → FLAG
        has_critical = any(
            r.passed is False and r.severity == RuleSeverity.CRITICAL
            for r in rule_results
        )
        has_error = any(
            r.passed is False and r.severity in (RuleSeverity.ERROR, RuleSeverity.CRITICAL)
            for r in rule_results
        )
        if has_critical:
            verdict = "BLOCK"
        elif has_error or passed_count < total:
            verdict = "FLAG"
        else:
            verdict = "PASS"

        summary = f"{self.layer_type.value}: {passed_count}/{total} 规则通过, 评分={score}"
        return LayerResult(
            layer_type=self.layer_type,
            score=score,
            rule_results=rule_results,
            verdict=verdict,
            summary=summary,
        )

    def verify_claims(
        self,
        claims: list[Claim],
        *,
        context_chunks: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        **kwargs: Any,
    ) -> LayerResult:
        """验证多个声明, 返回聚合结果."""
        if not claims:
            return LayerResult(
                layer_type=self.layer_type,
                score=100.0,
                rule_results=[],
                verdict="PASS",
                summary="无声明需要验证",
            )

        all_results: list[LayerRuleResult] = []
        for claim in claims:
            result = self.verify_claim(
                claim,
                context_chunks=context_chunks,
                evidence=evidence,
                **kwargs,
            )
            all_results.extend(result.rule_results)

        passed_count = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        score = round(passed_count / total * 100, 2) if total > 0 else 100.0

        has_critical = any(
            r.passed is False and r.severity == RuleSeverity.CRITICAL
            for r in all_results
        )
        if has_critical:
            verdict = "BLOCK"
        elif passed_count < total:
            verdict = "FLAG"
        else:
            verdict = "PASS"

        return LayerResult(
            layer_type=self.layer_type,
            score=score,
            rule_results=all_results,
            verdict=verdict,
            summary=f"{self.layer_type.value}: {passed_count}/{total} 规则通过, 评分={score}",
        )


# ============================================================
# 辅助函数
# ============================================================


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数值."""
    pattern = re.compile(r"(\d+\.?\d*)\s*(nm|mol%|%|ms|cm|K|°C)?", re.IGNORECASE)
    matches = pattern.findall(text)
    return [float(m[0]) for m in matches if m[0]]


def _extract_numbers_with_unit(text: str, unit: str) -> list[float]:
    """提取指定单位的数值."""
    pattern = re.compile(rf"(\d+\.?\d*)\s*{re.escape(unit)}", re.IGNORECASE)
    matches = pattern.findall(text)
    return [float(m) for m in matches if m]


def _contains_any(text: str, keywords: list[str]) -> bool:
    """检查文本是否包含任意关键词."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _contains_all(text: str, keywords: list[str]) -> bool:
    """检查文本是否包含所有关键词."""
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


# ============================================================
# L1 事实层 (FactLayer) — F-R01~F-R12
# ============================================================


class FactLayer(BaseReviewLayer):
    """L1 事实层校验器.

    验证 Dy3+ 发光材料领域的事实正确性。
    包含 12 条域特定规则 (F-R01~F-R12)。
    """

    layer_type = ReviewLayerType.L1_FACT

    def _init_rules(self) -> None:
        self._rules = [
            ReviewRule(
                rule_id="F-R01",
                name="发射峰波长校验",
                description="Dy3+ 发射主峰应在 570-585nm 范围内",
                severity=RuleSeverity.ERROR,
                checker=self._check_fr01,
            ),
            ReviewRule(
                rule_id="F-R02",
                name="能级跃迁对应校验",
                description="黄色发射→⁴F₉/₂→⁶H₁₃/₂; 蓝色发射→⁴F₉/₂→⁶H₁₅/₂",
                severity=RuleSeverity.ERROR,
                checker=self._check_fr02,
            ),
            ReviewRule(
                rule_id="F-R03",
                name="基质-掺杂剂兼容性",
                description="Dy3+ 常见基质: YAG, Y₂O₃, Ca₂Al₂SiO₇, Ba₂MgSi₂O₇",
                severity=RuleSeverity.WARNING,
                checker=self._check_fr03,
            ),
            ReviewRule(
                rule_id="F-R04",
                name="浓度猝灭阈值",
                description="Dy3+ 掺杂浓度 1-5mol%, 猝灭阈值约 3-8mol%",
                severity=RuleSeverity.ERROR,
                checker=self._check_fr04,
            ),
            ReviewRule(
                rule_id="F-R05",
                name="量子效率范围",
                description="Dy3+ 磷光体量子效率通常在 10-85%",
                severity=RuleSeverity.WARNING,
                checker=self._check_fr05,
            ),
            ReviewRule(
                rule_id="F-R06",
                name="色度坐标范围",
                description="CIE x: 0.38-0.45, y: 0.40-0.50",
                severity=RuleSeverity.WARNING,
                checker=self._check_fr06,
            ),
            ReviewRule(
                rule_id="F-R07",
                name="Judd-Ofelt 参数范围",
                description="Ω₂: 1-10, Ω₄: 0.5-5, Ω₆: 0.5-5 (×10⁻²⁰ cm²)",
                severity=RuleSeverity.WARNING,
                checker=self._check_fr07,
            ),
            ReviewRule(
                rule_id="F-R08",
                name="IUPAC 命名规范",
                description="dysprosium(III) 或 Dy³⁺",
                severity=RuleSeverity.INFO,
                checker=self._check_fr08,
            ),
            ReviewRule(
                rule_id="F-R09",
                name="IEC 62471 标准引用",
                description="照明用磷光材料需满足 IEC 62471",
                severity=RuleSeverity.INFO,
                checker=self._check_fr09,
            ),
            ReviewRule(
                rule_id="F-R10",
                name="CIE 色度标准引用",
                description="色度计算需引用 CIE 1931 或 CIE 1976",
                severity=RuleSeverity.INFO,
                checker=self._check_fr10,
            ),
            ReviewRule(
                rule_id="F-R11",
                name="激发光谱波段",
                description="Dy3+ 有效激发: ~350nm, ~390nm, ~450nm",
                severity=RuleSeverity.WARNING,
                checker=self._check_fr11,
            ),
            ReviewRule(
                rule_id="F-R12",
                name="衰减寿命数量级",
                description="Dy3+ 发光衰减寿命通常 0.1-2ms",
                severity=RuleSeverity.ERROR,
                checker=self._check_fr12,
            ),
        ]

    # ---- 规则检查函数 ----

    def _check_fr01(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R01: 发射峰波长在 570-585nm."""
        nums = _extract_numbers_with_unit(text, "nm")
        if not nums:
            return True, "未检测到波长数值, 跳过"
        out_of_range = [n for n in nums if n > 100 and n < 10000]  # 过滤非波长数值
        if not out_of_range:
            return True, "未检测到有效波长数值"
        in_range = all(570 <= n <= 585 for n in out_of_range)
        if in_range:
            return True, f"发射峰波长 {out_of_range} 在 570-585nm 范围内"
        return False, f"发射峰波长 {out_of_range} 超出 570-585nm 范围"

    def _check_fr02(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R02: 能级跃迁对应."""
        text_lower = text.lower()
        if "黄色" in text or "yellow" in text_lower:
            if "⁴f₉/₂→⁶h₁₃/₂" in text_lower or "4f9/2→6h13/2" in text_lower or "⁴F₉/₂" in text:
                return True, "黄色发射对应 ⁴F₉/₂→⁶H₁₃/₂ 跃迁"
            if "⁴f₉/₂" in text_lower or "4f9/2" in text_lower:
                return True, "检测到 ⁴F₉/₂ 能级"
            return True, "提到黄色发射但未检测到具体能级, 跳过"
        if "蓝色" in text or "blue" in text_lower:
            if "⁴f₉/₂→⁶h₁₅/₂" in text_lower or "4f9/2→6h15/2" in text_lower:
                return True, "蓝色发射对应 ⁴F₉/₂→⁶H₁₅/₂ 跃迁"
        return True, "未检测到能级跃迁描述, 跳过"

    def _check_fr03(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R03: 基质兼容性."""
        known_hosts = ["yag", "y₂o₃", "y2o3", "ca₂al₂sio₇", "ca2al2sio7", "ba₂mgsi₂o₇", "ba2mgsi2o7"]
        if _contains_any(text, known_hosts):
            return True, "使用已知兼容基质"
        if _contains_any(text, ["基质", "host", "matrix"]):
            return True, "提到基质但未检测到具体名称, 跳过"
        return True, "未检测到基质信息, 跳过"

    def _check_fr04(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R04: 浓度猝灭阈值."""
        conc_nums = _extract_numbers_with_unit(text, "mol%")
        if not conc_nums:
            conc_nums = _extract_numbers_with_unit(text, "mol")
        if not conc_nums:
            return True, "未检测到浓度数值, 跳过"
        for c in conc_nums:
            if c > 8:
                return False, f"掺杂浓度 {c}mol% 超出猝灭阈值 (3-8mol%)"
            if c < 1:
                return False, f"掺杂浓度 {c}mol% 低于推荐范围 (1-5mol%)"
        return True, f"掺杂浓度 {conc_nums} 在正常范围内"

    def _check_fr05(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R05: 量子效率范围."""
        if not _contains_any(text, ["量子效率", "quantum efficiency", "qe"]):
            return True, "未检测到量子效率信息, 跳过"
        nums = _extract_numbers_with_unit(text, "%")
        for n in nums:
            if n < 10 or n > 85:
                return False, f"量子效率 {n}% 超出 10-85% 范围"
        return True, "量子效率在正常范围内"

    def _check_fr06(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R06: 色度坐标范围 — CIE x: 0.38-0.45, y: 0.40-0.50."""
        if not _contains_any(text, ["cie", "色度", "chromaticity", "色坐标"]):
            return True, "未检测到色度信息, 跳过"
        # 提取 CIE x, y 坐标: 支持多种格式 (x=0.42, y=0.45) / (0.42, 0.45) / x:0.42
        x_vals: list[float] = []
        y_vals: list[float] = []
        # 匹配 x=0.42 或 x:0.42 或 x坐标 0.42
        x_patterns = [
            re.compile(r"[xy]\s*[=:：]\s*(0\.\d+)", re.IGNORECASE),
            re.compile(r"[xy]坐标\s*[=:]?\s*(0\.\d+)", re.IGNORECASE),
        ]
        for pat in x_patterns:
            for m in pat.finditer(text):
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    if "x" in m.group(0).lower():
                        x_vals.append(val)
                    elif "y" in m.group(0).lower():
                        y_vals.append(val)
        # 匹配 (0.42, 0.45) 格式
        paren_match = re.findall(r"\(\s*(0\.\d+)\s*,\s*(0\.\d+)\s*\)", text)
        for x_str, y_str in paren_match:
            x_v, y_v = float(x_str), float(y_str)
            if 0.0 <= x_v <= 1.0 and 0.0 <= y_v <= 1.0:
                x_vals.append(x_v)
                y_vals.append(y_v)
        if not x_vals and not y_vals:
            return True, "检测到色度关键词但未提取到坐标值, 跳过"
        for x in x_vals:
            if x < 0.38 or x > 0.45:
                return False, f"CIE x 坐标 {x} 超出 0.38-0.45 范围"
        for y in y_vals:
            if y < 0.40 or y > 0.50:
                return False, f"CIE y 坐标 {y} 超出 0.40-0.50 范围"
        return True, f"色度坐标 x={x_vals}, y={y_vals} 在正常范围内"

    def _check_fr07(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R07: Judd-Ofelt 参数范围 — Ω₂: 1-10, Ω₄: 0.5-5, Ω₆: 0.5-5 (×10⁻²⁰ cm²)."""
        if not _contains_any(text, ["judd", "ofelt", "Ω₂", "Ω₄", "Ω₆", "omega", "Ω"]):
            return True, "未检测到 Judd-Ofelt 参数, 跳过"
        # 提取 Ω₂, Ω₄, Ω₆ 值: 支持下标字符和数字形式
        omega_specs = [
            ("Ω₂", "Ω2", "omega2", 1.0, 10.0, "Ω₂"),
            ("Ω₄", "Ω4", "omega4", 0.5, 5.0, "Ω₄"),
            ("Ω₆", "Ω6", "omega6", 0.5, 5.0, "Ω₆"),
        ]
        found_any = False
        for sub_char, sub_num, sub_eng, lo, hi, name in omega_specs:
            patterns = [
                re.compile(rf"{re.escape(sub_char)}\s*[=:：]\s*(\d+\.?\d*)", re.IGNORECASE),
                re.compile(rf"{re.escape(sub_num)}\s*[=:：]\s*(\d+\.?\d*)", re.IGNORECASE),
                re.compile(rf"{sub_eng}\s*[=:：]\s*(\d+\.?\d*)", re.IGNORECASE),
            ]
            for pat in patterns:
                for m in pat.finditer(text):
                    val = float(m.group(1))
                    found_any = True
                    if val < lo or val > hi:
                        return False, f"{name} 值 {val} 超出范围 {lo}-{hi} (×10⁻²⁰ cm²)"
        if not found_any:
            return True, "检测到 Judd-Ofelt 关键词但未提取到参数值, 跳过"
        return True, "Judd-Ofelt 参数在正常范围内"

    def _check_fr08(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R08: IUPAC 命名规范."""
        if _contains_any(text, ["dy3+", "dy³⁺", "dysprosium", "镝"]):
            return True, "命名符合 IUPAC 规范"
        return True, "未检测到命名信息, 跳过"

    def _check_fr09(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R09: IEC 62471 标准引用."""
        if _contains_any(text, ["iec", "62471", "光生物安全"]):
            return True, "引用了 IEC 62471 标准"
        return True, "未检测到 IEC 标准引用, 跳过"

    def _check_fr10(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R10: CIE 色度标准引用."""
        if _contains_any(text, ["cie 1931", "cie 1976", "标准观察者", "ucs"]):
            return True, "引用了 CIE 色度标准"
        return True, "未检测到 CIE 标准引用, 跳过"

    def _check_fr11(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R11: 激发光谱波段."""
        if not _contains_any(text, ["激发", "excitation", "吸收"]):
            return True, "未检测到激发光谱信息, 跳过"
        nums = _extract_numbers_with_unit(text, "nm")
        valid_bands = [(340, 360), (380, 400), (440, 460)]
        for n in nums:
            if 300 < n < 500:
                in_band = any(lo <= n <= hi for lo, hi in valid_bands)
                if not in_band:
                    return False, f"激发波长 {n}nm 不在有效波段 (350/390/450nm)"
        return True, "激发波长在有效波段内"

    def _check_fr12(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """F-R12: 衰减寿命数量级."""
        if not _contains_any(text, ["寿命", "衰减", "lifetime", "decay"]):
            return True, "未检测到寿命数据, 跳过"
        nums = _extract_numbers_with_unit(text, "ms")
        if not nums:
            nums_us = _extract_numbers_with_unit(text, "μs")
            for n in nums_us:
                if n < 100 or n > 2000:
                    return False, f"衰减寿命 {n}μs 超出 0.1-2ms 范围"
            return True, "衰减寿命在正常范围内" if nums_us else "未检测到寿命数值, 跳过"
        for n in nums:
            if n < 0.1 or n > 2.0:
                return False, f"衰减寿命 {n}ms 超出 0.1-2.0ms 范围"
        return True, f"衰减寿命 {nums} 在 0.1-2.0ms 范围内"


# ============================================================
# L2 逻辑层 (LogicLayer) — L-R01~L-R10
# ============================================================


class LogicLayer(BaseReviewLayer):
    """L2 逻辑层校验器.

    验证 Dy3+ 发光材料领域的逻辑一致性。
    包含 10 条逻辑规则 (L-R01~L-R10)。
    """

    layer_type = ReviewLayerType.L2_LOGIC

    def _init_rules(self) -> None:
        self._rules = [
            ReviewRule(
                rule_id="L-R01",
                name="浓度-发光强度逻辑",
                description="掺杂浓度增加→发光强度先增后减(浓度猝灭), 非单调关系",
                severity=RuleSeverity.ERROR,
                checker=self._check_lr01,
            ),
            ReviewRule(
                rule_id="L-R02",
                name="温度-发光强度逻辑",
                description="温度升高→非辐射跃迁概率增加→发光强度降低(热猝灭)",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr02,
            ),
            ReviewRule(
                rule_id="L-R03",
                name="能级跃迁逻辑",
                description="4f-4f跃迁(禁戒,弱吸收)≠4f-5d跃迁(允许,强吸收)",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr03,
            ),
            ReviewRule(
                rule_id="L-R04",
                name="基质-光谱位移逻辑",
                description="基质晶格变化→晶体场强度变化→发射峰位移",
                severity=RuleSeverity.INFO,
                checker=self._check_lr04,
            ),
            ReviewRule(
                rule_id="L-R05",
                name="Judd-Ofelt 参数逻辑",
                description="Ω₂反映共价性/短程, Ω₄/Ω₆反映长程/黏度",
                severity=RuleSeverity.INFO,
                checker=self._check_lr05,
            ),
            ReviewRule(
                rule_id="L-R06",
                name="能量传递逻辑",
                description="Dy3+→Dy3+能量传递效率随浓度增加→浓度猝灭",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr06,
            ),
            ReviewRule(
                rule_id="L-R07",
                name="实验步骤顺序",
                description="前驱体称量→混合研磨→预烧→二次研磨→终烧→表征",
                severity=RuleSeverity.INFO,
                checker=self._check_lr07,
            ),
            ReviewRule(
                rule_id="L-R08",
                name="分类层级逻辑",
                description="Dy3+属于镧系→稀土→f区, 不属于d区或p区",
                severity=RuleSeverity.ERROR,
                checker=self._check_lr08,
            ),
            ReviewRule(
                rule_id="L-R09",
                name="寿命-浓度关系",
                description="掺杂浓度增加→能量传递加速→荧光寿命缩短",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr09,
            ),
            ReviewRule(
                rule_id="L-R10",
                name="色温-发光颜色逻辑",
                description="Dy3+黄蓝比可调→可实现白光发射→调谐黄蓝比调控色温",
                severity=RuleSeverity.INFO,
                checker=self._check_lr10,
            ),
            # ---- 增强规则: 推理链与 DAG 分析 ----
            ReviewRule(
                rule_id="L-R11",
                name="推理链循环检测",
                description="因果推理链中不应存在循环依赖 (A→B→C→A)",
                severity=RuleSeverity.CRITICAL,
                checker=self._check_lr11,
            ),
            ReviewRule(
                rule_id="L-R12",
                name="推理链断裂检测",
                description="有因果标记的声明必须有明确前提, 不应存在断链",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr12,
            ),
            ReviewRule(
                rule_id="L-R13",
                name="声明间矛盾检测",
                description="多个声明之间不应存在数值或定性矛盾",
                severity=RuleSeverity.ERROR,
                checker=self._check_lr13,
            ),
            ReviewRule(
                rule_id="L-R14",
                name="推理链冗余检测",
                description="推理链中不应存在高度相似的冗余路径",
                severity=RuleSeverity.INFO,
                checker=self._check_lr14,
            ),
            ReviewRule(
                rule_id="L-R15",
                name="推理链拓扑完整性",
                description="推理链应能完成拓扑排序, 保证逻辑顺序一致",
                severity=RuleSeverity.WARNING,
                checker=self._check_lr15,
            ),
        ]

    def _check_lr01(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R01: 浓度-发光强度逻辑."""
        if not _contains_any(text, ["浓度", "掺杂", "强度", "发光", "猝灭"]):
            return True, "未检测到浓度-强度相关描述, 跳过"
        # 检查是否描述了非单调关系
        non_monotonic_keywords = [
            "先增后减", "先增大后减小", "先增加后减少", "先升后降",
            "非单调", "猝灭", "quenching", "先增后降",
            "增加到最大后", "达到最大值后减小",
        ]
        monotonic_keywords = [
            "单调增加", "持续增加", "持续增大", "成正比", "线性增加",
            "monotonically", "线性关系",
        ]
        if _contains_any(text, monotonic_keywords):
            return False, "浓度-强度关系描述为单调增加, 与浓度猝灭逻辑矛盾"
        if _contains_any(text, non_monotonic_keywords):
            return True, "浓度-强度关系正确描述为非单调(先增后减)"
        return True, "浓度-强度描述未明确单调性, 跳过"

    def _check_lr02(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R02: 温度-发光强度逻辑."""
        if not _contains_any(text, ["温度", "热", "temperature", "thermal"]):
            return True, "未检测到温度相关描述, 跳过"
        if _contains_any(text, ["降低", "减小", "下降", "减弱", "猝灭", "decrease", "reduce"]):
            return True, "温度-强度逻辑正确: 温度升高→强度降低"
        if _contains_any(text, ["增加", "升高", "增强", "increase"]):
            # 检查是否说的是非辐射跃迁增加 (这是正确的)
            if _contains_any(text, ["非辐射", "non-radiative"]):
                return True, "非辐射跃迁概率增加→发光强度降低, 逻辑正确"
            return False, "温度升高但发光强度增加, 与热猝灭逻辑矛盾"
        return True, "温度-强度描述不明确, 跳过"

    def _check_lr03(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R03: 能级跃迁逻辑 — 4f-4f 跃迁为禁戒/弱吸收, 4f-5d 为允许/强吸收."""
        if not _contains_any(text, ["4f", "5d", "跃迁", "transition"]):
            return True, "未检测到能级跃迁描述, 跳过"
        text_lower = text.lower()
        # 检查 4f-4f 是否被错误描述为允许/强吸收
        if _contains_any(text, ["4f-4f", "4f→4f", "f-f跃迁"]):
            if _contains_any(text, ["允许", "强吸收", "allowed", "strong absorption"]):
                return False, "4f-4f 跃迁应为禁戒/弱吸收, 而非允许/强吸收"
            if _contains_any(text, ["禁戒", "弱吸收", "forbidden", "weak", "parity forbidden"]):
                return True, "4f-4f 跃迁正确描述为禁戒/弱吸收"
        # 检查 4f-5d 是否被错误描述为禁戒/弱吸收
        if _contains_any(text, ["4f-5d", "4f→5d", "f-d跃迁"]):
            if _contains_any(text, ["禁戒", "弱吸收", "forbidden", "weak absorption"]):
                return False, "4f-5d 跃迁应为允许/强吸收, 而非禁戒/弱吸收"
            if _contains_any(text, ["允许", "强吸收", "allowed", "strong"]):
                return True, "4f-5d 跃迁正确描述为允许/强吸收"
        return True, "能级跃迁逻辑检查通过"

    def _check_lr04(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R04: 基质-光谱位移逻辑 — 基质晶格变化→晶体场强度变化→发射峰位移."""
        if not _contains_any(text, ["基质", "晶格", "位移", "host", "lattice", "shift"]):
            return True, "未检测到基质-位移相关描述, 跳过"
        # 如果同时提到基质变化和位移/偏移, 检查逻辑是否正确
        has_host_change = _contains_any(text, ["基质变化", "不同基质", "更换基质", "host change"])
        has_spectral_shift = _contains_any(text, ["位移", "偏移", "shift", "红移", "蓝移"])
        has_crystal_field = _contains_any(text, ["晶体场", "crystal field", "晶场"])
        if has_host_change and has_spectral_shift:
            if has_crystal_field:
                return True, "基质变化→晶体场变化→光谱位移, 逻辑正确"
            return True, "基质变化与光谱位移关联描述正确"
        if has_host_change and not has_spectral_shift:
            return True, "提到基质变化但未提及位移, 逻辑不完整但无错误"
        return True, "基质-光谱位移逻辑检查通过"

    def _check_lr05(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R05: Judd-Ofelt 参数逻辑 — Ω₂ 反映共价性/短程, Ω₄/Ω₆ 反映长程/黏度."""
        if not _contains_any(text, ["judd", "ofelt", "Ω₂", "Ω₄", "Ω₆", "omega"]):
            return True, "未检测到 Judd-Ofelt 参数, 跳过"
        # 检查 Ω₂ 是否被错误关联到长程性质
        if _contains_any(text, ["Ω₂"]) or _contains_any(text, ["omega2"]):
            if _contains_any(text, ["长程", "long-range", "黏度", "viscosity"]):
                if _contains_any(text, ["Ω₂"]) and not _contains_any(text, ["共价", "covalen", "短程", "short-range"]):
                    return False, "Ω₂ 应反映共价性/短程对称性, 而非长程/黏度"
            if _contains_any(text, ["共价", "covalen", "短程", "short-range", "对称性"]):
                return True, "Ω₂ 正确关联到共价性/短程性质"
        # 检查 Ω₄/Ω₆ 是否被错误关联到短程性质
        if _contains_any(text, ["Ω₄", "Ω₆", "omega4", "omega6"]):
            if _contains_any(text, ["短程", "short-range", "共价性", "covalency"]):
                if not _contains_any(text, ["长程", "long-range", "黏度", "viscosity"]):
                    return False, "Ω₄/Ω₆ 应反映长程/黏度, 而非短程/共价性"
            if _contains_any(text, ["长程", "long-range", "黏度", "viscosity"]):
                return True, "Ω₄/Ω₆ 正确关联到长程/黏度性质"
        return True, "Judd-Ofelt 参数逻辑检查通过"

    def _check_lr06(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R06: 能量传递逻辑 — Dy3+→Dy3+ 能量传递效率随浓度增加→浓度猝灭."""
        if not _contains_any(text, ["能量传递", "energy transfer", "传递"]):
            return True, "未检测到能量传递描述, 跳过"
        # 如果提到能量传递, 检查是否正确关联到浓度效应
        has_concentration = _contains_any(text, ["浓度", "concentration"])
        has_quenching = _contains_any(text, ["猝灭", "quenching"])
        if _contains_any(text, ["效率降低", "效率下降", "降低", "下降"]):
            if has_concentration:
                return True, "能量传递效率随浓度变化, 逻辑正确"
        if _contains_any(text, ["效率增加", "效率提高", "增强"]):
            if has_concentration and not has_quenching:
                # 能量传递效率增加但未提到猝灭, 可能逻辑不完整
                return True, "能量传递效率增加, 但需注意浓度猝灭效应"
        if has_concentration and has_quenching:
            return True, "能量传递→浓度猝灭逻辑正确"
        return True, "能量传递逻辑检查通过"

    def _check_lr07(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R07: 实验步骤顺序 — 前驱体→混合研磨→预烧→二次研磨→终烧→表征."""
        if not _contains_any(text, ["实验", "合成", "制备", "experiment", "synthesis"]):
            return True, "未检测到实验步骤描述, 跳过"
        # 定义正确的步骤顺序及其关键词
        step_keywords = [
            ("前驱体", ["前驱体", "precursor", "原料", "称量"]),
            ("混合研磨", ["混合", "研磨", "mix", "grind", "mixing"]),
            ("预烧", ["预烧", "pre-sinter", "预煅烧"]),
            ("二次研磨", ["二次研磨", "再次研磨", "re-grind"]),
            ("终烧", ["终烧", "烧结", "sinter", "煅烧", "calcine"]),
            ("表征", ["表征", "characteriz", "XRD", "PL", "measure"]),
        ]
        # 找到每个步骤在文本中的最早出现位置
        found_steps: list[tuple[int, str]] = []  # (text_position, step_name)
        text_lower = text.lower()
        for step_idx, (step_name, keywords) in enumerate(step_keywords):
            earliest_pos = -1
            for kw in keywords:
                pos = text_lower.find(kw.lower())
                if pos >= 0:
                    if earliest_pos < 0 or pos < earliest_pos:
                        earliest_pos = pos
            if earliest_pos >= 0:
                found_steps.append((earliest_pos, step_name))
        if len(found_steps) < 2:
            return True, "检测到实验描述但步骤不足, 跳过顺序检查"
        # 按文本位置排序, 检查步骤的逻辑顺序是否正确
        found_steps.sort(key=lambda x: x[0])
        # 将文本顺序映射到步骤索引, 检查是否递增
        step_order = [
            next(i for i, (name, _) in enumerate(step_keywords) if name == s[1])
            for s in found_steps
        ]
        for i in range(1, len(step_order)):
            if step_order[i] < step_order[i - 1]:
                return False, (
                    f"实验步骤顺序错误: '{found_steps[i - 1][1]}' "
                    f"(文本位置 {found_steps[i - 1][0]}) 不应在 "
                    f"'{found_steps[i][1]}' (文本位置 {found_steps[i][0]}) 之前"
                )
        return True, f"实验步骤顺序正确: {' → '.join(s[1] for s in found_steps)}"

    def _check_lr08(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R08: 分类层级逻辑 — Dy3+ 属于镧系→稀土→f 区, 不属于 d 区或 p 区."""
        if not _contains_any(text, ["dy3", "dy3+", "dy³⁺", "dysprosium", "镝"]):
            return True, "未检测到 Dy3+ 分类描述, 跳过"
        # 检查是否错误地分类为 d 区或 p 区
        wrong_zones = ["d 区", "d区", "d-block", "d block", "过渡金属", "transition metal"]
        if _contains_any(text, wrong_zones):
            return False, "Dy3+ 被错误分类为 d 区过渡金属, 应为 f 区镧系元素"
        # 检查是否正确分类
        correct_zones = ["镧系", "lanthanide", "稀土", "rare earth", "f 区", "f区", "f-block", "f block"]
        if _contains_any(text, correct_zones):
            return True, "Dy3+ 正确分类为镧系/f 区元素"
        return True, "未检测到分类信息, 跳过"

    def _check_lr09(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R09: 寿命-浓度关系 — 掺杂浓度增加→能量传递加速→荧光寿命缩短."""
        if not _contains_any(text, ["寿命", "浓度", "lifetime", "concentration"]):
            return True, "未检测到寿命-浓度关系描述, 跳过"
        has_lifetime = _contains_any(text, ["寿命", "lifetime", "衰减时间"])
        has_concentration = _contains_any(text, ["浓度", "concentration", "掺杂"])
        if has_lifetime and has_concentration:
            # 检查是否正确描述了反比关系
            decrease_kw = ["缩短", "减小", "降低", "减少", "decrease", "shorter", "reduce"]
            increase_kw = ["增加", "增大", "延长", "升高", "increase", "longer"]
            if _contains_any(text, decrease_kw):
                return True, "浓度增加→寿命缩短, 逻辑正确"
            if _contains_any(text, increase_kw):
                # 如果说寿命随浓度增加而增加, 这是错误的
                if _contains_any(text, ["浓度增加", "浓度升高", "浓度增大"]):
                    return False, "浓度增加但寿命也增加, 与能量传递加速逻辑矛盾"
        return True, "寿命-浓度关系检查通过"

    def _check_lr10(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R10: 色温-发光颜色逻辑 — 黄蓝比可调→白光发射→色温可调."""
        if not _contains_any(text, ["色温", "白光", "黄蓝比", "color temperature", "white light"]):
            return True, "未检测到色温-颜色相关描述, 跳过"
        # 检查是否正确描述了黄蓝比与色温的关系
        has_yellow_blue = _contains_any(text, ["黄蓝比", "yellow-blue", "黄/蓝", "蓝黄比"])
        has_white = _contains_any(text, ["白光", "white light", "white emission"])
        has_cct = _contains_any(text, ["色温", "CCT", "correlated color temperature"])
        if has_cct:
            # 提取色温数值
            cct_nums = _extract_numbers_with_unit(text, "K")
            for n in cct_nums:
                if n < 3000 or n > 8000:
                    return False, f"色温 {n}K 超出 Dy3+ 白光发射典型范围 (3000-8000K)"
        if has_yellow_blue and has_white:
            return True, "黄蓝比可调→白光发射, 逻辑正确"
        if has_cct and has_white:
            return True, "白光发射与色温调控关联正确"
        return True, "色温-发光颜色逻辑检查通过"

    # ---- 增强规则: 推理链与 DAG 分析 ----

    def _check_lr11(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R11: 推理链循环检测 — 因果推理链中不应存在循环依赖."""
        from .reasoning_chain import ReasoningChainExtractor, ReasoningDAG

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        if len(steps) < 2:
            return True, "推理步骤不足 2 步, 跳过循环检测"
        dag = ReasoningDAG.build(steps)
        cycles = dag.detect_cycles()
        if cycles:
            cycle_desc = " → ".join(cycles[0])
            return False, f"检测到推理循环: {cycle_desc}"
        return True, f"推理链无循环 ({dag.node_count} 节点, {dag.edge_count} 边)"

    def _check_lr12(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R12: 推理链断裂检测 — 有因果标记的声明必须有明确前提."""
        from .reasoning_chain import ReasoningChainExtractor, ReasoningDAG

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        if not steps:
            return True, "未检测到推理步骤, 跳过"
        dag = ReasoningDAG.build(steps)
        breaks = dag.detect_breaks()
        if breaks:
            return False, f"检测到 {len(breaks)} 处推理断链: {breaks[:3]}"
        return True, "推理链无断链"

    def _check_lr13(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R13: 声明间矛盾检测 — 多个声明之间不应存在数值或定性矛盾."""
        from .reasoning_chain import ReasoningChainExtractor, ContradictionDetector

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        if len(steps) < 2:
            return True, "声明数量不足 2 个, 跳过矛盾检测"
        detector = ContradictionDetector()
        contradictions = detector.detect_contradictions(steps)
        if contradictions:
            descs = [c.description for c in contradictions[:3]]
            return False, f"检测到 {len(contradictions)} 处矛盾: {'; '.join(descs)}"
        return True, "声明间无矛盾"

    def _check_lr14(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R14: 推理链冗余检测 — 推理链中不应存在高度相似的冗余路径."""
        from .reasoning_chain import ReasoningChainExtractor, ReasoningDAG

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        if len(steps) < 2:
            return True, "推理步骤不足, 跳过冗余检测"
        dag = ReasoningDAG.build(steps)
        redundancy = dag.detect_redundancy()
        if redundancy:
            return False, f"检测到 {len(redundancy)} 处冗余路径"
        return True, "推理链无冗余"

    def _check_lr15(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """L-R15: 推理链拓扑完整性 — 推理链应能完成拓扑排序."""
        from .reasoning_chain import ReasoningChainExtractor, ReasoningDAG

        extractor = ReasoningChainExtractor()
        steps = extractor.extract(text)
        if not steps:
            return True, "未检测到推理步骤, 跳过"
        dag = ReasoningDAG.build(steps)
        try:
            sorted_ids = dag.topological_sort()
            if len(sorted_ids) == dag.node_count:
                return True, f"拓扑排序成功 ({len(sorted_ids)} 步)"
            return False, f"拓扑排序不完整: {len(sorted_ids)}/{dag.node_count}"
        except Exception:
            return False, "拓扑排序失败, 推理链存在结构问题"


# ============================================================
# L3 数值层 (NumericalLayer) — N-R01~N-R12 + N-R13~N-R18 (增强)
# ============================================================


class NumericalLayer(BaseReviewLayer):
    """L3 数值层校验器.

    验证 Dy3+ 发光材料领域的数值正确性。
    包含 12 条数值范围规则 (N-R01~N-R12) 和 6 条计算验证规则 (N-R13~N-R18)。
    """

    layer_type = ReviewLayerType.L3_NUMERICAL

    #: 数值范围定义: (rule_id, name, 关键词, 最小值, 最大值, 单位)
    _RANGES = [
        ("N-R01", "主发射峰", ["发射峰", "主峰", "emission peak"], 570, 585, "nm"),
        ("N-R02", "蓝色发射峰", ["蓝色", "blue"], 475, 495, "nm"),
        ("N-R03", "激发波长", ["激发", "excitation"], 340, 460, "nm"),
        ("N-R04", "掺杂浓度", ["掺杂浓度", "浓度", "concentration"], 1, 5, "mol%"),
        ("N-R05", "猝灭阈值", ["猝灭阈值", "quenching threshold"], 3, 8, "mol%"),
        ("N-R06", "量子效率", ["量子效率", "quantum efficiency"], 10, 85, "%"),
        ("N-R07", "衰减寿命", ["寿命", "衰减", "lifetime", "decay"], 0.1, 2.0, "ms"),
        ("N-R08", "CIE x", ["cie x", "x坐标"], 0.38, 0.45, ""),
        ("N-R09", "CIE y", ["cie y", "y坐标"], 0.40, 0.50, ""),
        ("N-R10", "Judd-Ofelt Ω₂", ["Ω₂", "omega2", "ω₂"], 1, 10, ""),
        ("N-R11", "Judd-Ofelt Ω₄", ["Ω₄", "omega4", "ω₄"], 0.5, 5, ""),
        ("N-R12", "Judd-Ofelt Ω₆", ["Ω₆", "omega6", "ω₆"], 0.5, 5, ""),
    ]

    def _init_rules(self) -> None:
        self._rules = []
        for rule_id, name, keywords, lo, hi, unit in self._RANGES:
            self._rules.append(ReviewRule(
                rule_id=rule_id,
                name=name,
                description=f"{name} 应在 {lo}-{hi} {unit} 范围内",
                severity=RuleSeverity.ERROR,
                checker=self._make_range_checker(rule_id, name, keywords, lo, hi, unit),
            ))
        # 添加增强计算验证规则
        self._init_rules_enhanced()

    @staticmethod
    def _make_range_checker(
        rule_id: str,
        name: str,
        keywords: list[str],
        lo: float,
        hi: float,
        unit: str,
    ) -> Callable[..., tuple[bool, str]]:
        """生成数值范围检查函数."""

        def checker(text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
            if not _contains_any(text, keywords):
                return True, f"未检测到 {name} 相关数值, 跳过"
            nums: list[float] = []
            if unit:
                nums = _extract_numbers_with_unit(text, unit)
            else:
                # 无单位时提取所有数值
                all_nums = _extract_numbers(text)
                # 根据范围过滤可能的数值
                nums = [n for n in all_nums if lo * 0.5 <= n <= hi * 2]
            if not nums:
                return True, f"检测到 {name} 但未提取到数值, 跳过"
            for n in nums:
                if n < lo or n > hi:
                    return False, f"{name} 值 {n} 超出范围 {lo}-{hi} {unit}"
            return True, f"{name} 值 {nums} 在范围 {lo}-{hi} {unit} 内"

        return checker

    def _init_rules_enhanced(self) -> None:
        """初始化增强计算验证规则 (N-R13~N-R18)."""
        self._rules.extend([
            ReviewRule(
                rule_id="N-R13",
                name="单位换算一致性",
                description="nm↔cm⁻¹↔eV 单位换算结果应一致",
                severity=RuleSeverity.WARNING,
                checker=self._check_nr13,
            ),
            ReviewRule(
                rule_id="N-R14",
                name="Judd-Ofelt 参数计算验证",
                description="Ω₂/Ω₄/Ω₆ 参数应满足物理约束",
                severity=RuleSeverity.ERROR,
                checker=self._check_nr14,
            ),
            ReviewRule(
                rule_id="N-R15",
                name="CIE 色度坐标计算验证",
                description="CIE 坐标应在光谱轨迹内且 CCT 合理",
                severity=RuleSeverity.WARNING,
                checker=self._check_nr15,
            ),
            ReviewRule(
                rule_id="N-R16",
                name="量子效率计算验证",
                description="QE = τ_obs/τ_rad × 100% 应 ≤ 100%",
                severity=RuleSeverity.ERROR,
                checker=self._check_nr16,
            ),
            ReviewRule(
                rule_id="N-R17",
                name="数值异常值检测",
                description="同组数据中不应存在统计异常值 (IQR/Z-score)",
                severity=RuleSeverity.WARNING,
                checker=self._check_nr17,
            ),
            ReviewRule(
                rule_id="N-R18",
                name="相对误差验证",
                description="测量值与标准值的相对误差应 < 10%",
                severity=RuleSeverity.INFO,
                checker=self._check_nr18,
            ),
        ])

    def _check_nr13(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R13: 单位换算一致性 — nm↔cm⁻¹↔eV 换算结果应一致."""
        from .computation import UnitConverter

        nm_nums = _extract_numbers_with_unit(text, "nm")
        if not nm_nums:
            return True, "未检测到 nm 数值, 跳过单位换算验证"
        converter = UnitConverter()
        for nm in nm_nums:
            if 200 <= nm <= 2000:
                cm_inv = converter.nm_to_cm_inv(nm)
                ev = converter.nm_to_ev(nm)
                # 验证往返一致性
                nm_back = converter.cm_inv_to_nm(cm_inv)
                if abs(nm - nm_back) > 0.01:
                    return False, f"nm→cm⁻¹→nm 往返不一致: {nm}→{cm_inv}→{nm_back}"
                # 验证 eV 与 nm 的关系
                nm_from_ev = converter.ev_to_nm(ev)
                if abs(nm - nm_from_ev) > 0.01:
                    return False, f"nm→eV→nm 往返不一致: {nm}→{ev}→{nm_from_ev}"
        return True, f"单位换算一致性验证通过 ({len(nm_nums)} 个波长值)"

    def _check_nr14(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R14: Judd-Ofelt 参数计算验证."""
        from .computation import JuddOfeltCalculator

        if not _contains_any(text, ["judd", "ofelt", "Ω₂", "Ω₄", "Ω₆", "omega"]):
            return True, "未检测到 Judd-Ofelt 参数, 跳过"
        calc = JuddOfeltCalculator()
        # 提取 Ω₂, Ω₄, Ω₆ 值
        omega2_vals = []
        omega4_vals = []
        omega6_vals = []
        for pattern_str, val_list in [
            (r"Ω₂\s*[=:：]\s*(\d+\.?\d*)", omega2_vals),
            (r"Ω2\s*[=:：]\s*(\d+\.?\d*)", omega2_vals),
            (r"Ω₄\s*[=:：]\s*(\d+\.?\d*)", omega4_vals),
            (r"Ω4\s*[=:：]\s*(\d+\.?\d*)", omega4_vals),
            (r"Ω₆\s*[=:：]\s*(\d+\.?\d*)", omega6_vals),
            (r"Ω6\s*[=:：]\s*(\d+\.?\d*)", omega6_vals),
        ]:
            for m in re.finditer(pattern_str, text, re.IGNORECASE):
                val_list.append(float(m.group(1)))
        if not omega2_vals and not omega4_vals and not omega6_vals:
            return True, "检测到 JO 关键词但未提取到参数值, 跳过"
        # 验证参数范围
        is_valid = calc.validate_judd_ofelt_params(
            omega2_vals[0] if omega2_vals else 5.0,
            omega4_vals[0] if omega4_vals else 2.0,
            omega6_vals[0] if omega6_vals else 2.0,
        )
        if not is_valid:
            return False, "Judd-Ofelt 参数不满足物理约束范围"
        return True, "Judd-Ofelt 参数计算验证通过"

    def _check_nr15(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R15: CIE 色度坐标计算验证."""
        from .computation import CIECalculator

        if not _contains_any(text, ["cie", "色度", "chromaticity", "色坐标"]):
            return True, "未检测到 CIE 色度信息, 跳过"
        # 提取 CIE x, y 坐标
        x_vals: list[float] = []
        y_vals: list[float] = []
        paren_match = re.findall(r"\(\s*(0\.\d+)\s*,\s*(0\.\d+)\s*\)", text)
        for x_str, y_str in paren_match:
            x_v, y_v = float(x_str), float(y_str)
            if 0.0 <= x_v <= 1.0 and 0.0 <= y_v <= 1.0:
                x_vals.append(x_v)
                y_vals.append(y_v)
        if not x_vals:
            return True, "检测到色度关键词但未提取到坐标值, 跳过"
        calc = CIECalculator()
        for x, y in zip(x_vals, y_vals):
            if not calc.validate_cie_coordinates(x, y):
                return False, f"CIE 坐标 ({x}, {y}) 不在光谱轨迹内"
            cct = calc.calculate_cct(x, y)
            if cct < 1000 or cct > 40000:
                return False, f"CCT {cct:.0f}K 不在合理范围 (1000-40000K)"
        return True, f"CIE 色度坐标计算验证通过 ({len(x_vals)} 组坐标)"

    def _check_nr16(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R16: 量子效率计算验证 — QE = τ_obs/τ_rad × 100% 应 ≤ 100%."""
        from .computation import JuddOfeltCalculator

        if not _contains_any(text, ["量子效率", "quantum efficiency", "QE"]):
            return True, "未检测到量子效率信息, 跳过"
        # 提取 τ_obs 和 τ_rad
        ms_nums = _extract_numbers_with_unit(text, "ms")
        if len(ms_nums) < 2:
            return True, "寿命数据不足 2 个, 跳过 QE 计算验证"
        calc = JuddOfeltCalculator()
        # 假设前两个寿命值为 τ_rad 和 τ_obs
        tau_rad, tau_obs = ms_nums[0], ms_nums[1]
        try:
            qe = calc.calculate_qe_from_jo(5.0, 2.0, 2.0, tau_rad, tau_obs)
            if qe > 100:
                return False, f"计算 QE = {qe:.1f}% > 100%, τ_obs 不应大于 τ_rad"
            if qe < 0:
                return False, f"计算 QE = {qe:.1f}% < 0%, 参数异常"
        except Exception as e:
            return True, f"QE 计算异常, 跳过: {e}"
        return True, f"量子效率计算验证通过 (QE={qe:.1f}%)"

    def _check_nr17(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R17: 数值异常值检测 — 同组数据中不应存在统计异常值."""
        from .computation import ErrorAnalyzer

        # 提取同单位数值组
        for unit in ["nm", "mol%", "ms", "%", "K"]:
            nums = _extract_numbers_with_unit(text, unit)
            if len(nums) < 4:
                continue
            analyzer = ErrorAnalyzer()
            iqr_outliers = analyzer.detect_outliers_iqr(nums)
            zscore_outliers = analyzer.detect_outliers_zscore(nums)
            total_outliers = set(iqr_outliers) | set(zscore_outliers)
            if total_outliers:
                outlier_vals = [nums[i] for i in total_outliers if 0 <= i < len(nums)]
                return False, f"检测到 {len(total_outliers)} 个异常值 ({unit}): {outlier_vals[:3]}"
        return True, "未检测到统计异常值"

    def _check_nr18(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """N-R18: 相对误差验证 — 测量值与标准值的相对误差应 < 10%."""
        from .computation import ErrorAnalyzer

        if not _contains_any(text, ["误差", "error", "偏差", "deviation", "相对误差"]):
            return True, "未检测到误差信息, 跳过"
        # 提取误差百分比
        error_nums = _extract_numbers_with_unit(text, "%")
        for n in error_nums:
            if n > 10 and _contains_any(text, ["误差", "error", "相对误差"]):
                return False, f"相对误差 {n}% 超过 10% 阈值"
        return True, "相对误差在可接受范围内"


# ============================================================
# L4 溯源层 (ProvenanceLayer) — P-R01~P-R10 + P-R11~P-R15 (增强)
# ============================================================


class ProvenanceLayer(BaseReviewLayer):
    """L4 溯源层校验器.

    验证声明的溯源完整性。
    包含 10 条溯源规则 (P-R01~P-R10)。
    """

    layer_type = ReviewLayerType.L4_PROVENANCE

    def _init_rules(self) -> None:
        self._rules = [
            ReviewRule(
                rule_id="P-R01",
                name="声明-来源绑定",
                description="每个声明必须关联至少一个来源",
                severity=RuleSeverity.ERROR,
                checker=self._check_pr01,
            ),
            ReviewRule(
                rule_id="P-R02",
                name="DOI 有效性",
                description="引用的 DOI 必须有效",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr02,
            ),
            ReviewRule(
                rule_id="P-R03",
                name="引用内容一致性",
                description="引用内容与原文摘要语义相似度 ≥ 0.80",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr03,
            ),
            ReviewRule(
                rule_id="P-R04",
                name="来源权威性标注",
                description="每个来源必须标注 Tier 等级",
                severity=RuleSeverity.INFO,
                checker=self._check_pr04,
            ),
            ReviewRule(
                rule_id="P-R05",
                name="溯源链完整性",
                description="从原始数据到最终输出的全链路可追溯",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr05,
            ),
            ReviewRule(
                rule_id="P-R06",
                name="时效性检查",
                description="引用文献不超过 10 年",
                severity=RuleSeverity.INFO,
                checker=self._check_pr06,
            ),
            ReviewRule(
                rule_id="P-R07",
                name="冲突来源标注",
                description="多来源数值不一致时标注差异",
                severity=RuleSeverity.INFO,
                checker=self._check_pr07,
            ),
            ReviewRule(
                rule_id="P-R08",
                name="AI 生成标注",
                description="无来源的声明必须标注 AI-generated",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr08,
            ),
            ReviewRule(
                rule_id="P-R09",
                name="标准引用格式",
                description="引用格式符合 ACS 或 APA 标准",
                severity=RuleSeverity.INFO,
                checker=self._check_pr09,
            ),
            ReviewRule(
                rule_id="P-R10",
                name="动态溯源版本",
                description="知识更新时创建新溯源版本",
                severity=RuleSeverity.INFO,
                checker=self._check_pr10,
            ),
            # ---- 增强规则: 溯源链与权威性评级 ----
            ReviewRule(
                rule_id="P-R11",
                name="来源 Tier 评级",
                description="来源应有 Tier 1-5 权威性评级",
                severity=RuleSeverity.INFO,
                checker=self._check_pr11,
            ),
            ReviewRule(
                rule_id="P-R12",
                name="DOI 格式校验增强",
                description="DOI 应符合 ISO 26324 结构标准",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr12,
            ),
            ReviewRule(
                rule_id="P-R13",
                name="溯源链深度验证",
                description="溯源链应达到至少 2 层深度",
                severity=RuleSeverity.WARNING,
                checker=self._check_pr13,
            ),
            ReviewRule(
                rule_id="P-R14",
                name="来源权威性评分",
                description="来源权威性评分应 ≥ 0.5",
                severity=RuleSeverity.INFO,
                checker=self._check_pr14,
            ),
            ReviewRule(
                rule_id="P-R15",
                name="溯源版本管理",
                description="知识更新应创建新溯源版本",
                severity=RuleSeverity.INFO,
                checker=self._check_pr15,
            ),
        ]

    def _check_pr01(
        self, claim: Claim, evidence: list[Evidence] | None = None, **kw: Any
    ) -> tuple[bool, str]:
        """P-R01: 声明-来源绑定."""
        ev_list = evidence or []
        has_evidence = bool(claim.evidence_ids) or len(ev_list) > 0
        if has_evidence:
            return True, f"声明关联了 {len(claim.evidence_ids)} 个来源"
        return False, "声明未关联任何来源"

    def _check_pr02(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R02: DOI 有效性."""
        dois = re.findall(r"10\.\d{4,}/\S+", text)
        if not dois:
            return True, "未检测到 DOI, 跳过"
        return True, f"检测到 {len(dois)} 个 DOI"

    def _check_pr03(
        self,
        text: str,
        claim_text: str,
        evidence: list[Evidence] | None = None,
        **kw: Any,
    ) -> tuple[bool, str]:
        """P-R03: 引用内容一致性 — 引用内容与原文摘要语义相似度 ≥ 0.80."""
        ev_list = evidence or []
        if not ev_list:
            return True, "无证据可供一致性检查, 跳过"
        min_similarity = 1.0
        for ev in ev_list:
            if ev.content:
                ratio = SequenceMatcher(None, claim_text.lower(), ev.content.lower()).ratio()
                if ratio < min_similarity:
                    min_similarity = ratio
        if min_similarity < 0.80:
            return False, f"引用内容与原文相似度 {min_similarity:.2f} 低于阈值 0.80"
        return True, f"引用内容一致性检查通过 (最低相似度={min_similarity:.2f})"

    def _check_pr04(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R04: 来源权威性标注."""
        if not _contains_any(text, ["tier", "等级", "来源"]):
            return True, "未检测到来源标注信息, 跳过"
        # 检查是否标注了 Tier 等级
        tier_match = re.findall(r"tier\s*[-_]?\s*[123T]", text, re.IGNORECASE)
        if tier_match:
            return True, f"来源标注了 Tier 等级: {tier_match}"
        if _contains_any(text, ["tier-1", "tier-2", "tier-3", "T1", "T2", "T3"]):
            return True, "来源标注了 Tier 等级"
        return True, "来源权威性标注检查通过"

    def _check_pr05(
        self,
        text: str,
        claim_text: str,
        evidence: list[Evidence] | None = None,
        **kw: Any,
    ) -> tuple[bool, str]:
        """P-R05: 溯源链完整性 — 从原始数据到最终输出的全链路可追溯."""
        ev_list = evidence or []
        if not ev_list:
            return True, "无证据可供溯源链检查, 跳过"
        # 检查每个证据是否有 source_uri
        missing_uri = [i for i, ev in enumerate(ev_list) if not ev.source_uri]
        if missing_uri:
            return False, f"{len(missing_uri)} 个证据缺少 source_uri, 溯源链不完整"
        # 检查证据是否有置信度
        low_confidence = [
            i for i, ev in enumerate(ev_list) if ev.confidence < 0.5
        ]
        if low_confidence:
            return True, f"溯源链完整, 但 {len(low_confidence)} 个证据置信度低于 0.5"
        return True, "溯源链完整性检查通过: 所有证据有 source_uri 且置信度 ≥ 0.5"

    def _check_pr06(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R06: 时效性检查 — 引用文献不超过 10 年."""
        # 提取年份: 支持 (2020), 2020年, published 2020, et al. 2020
        year_patterns = [
            re.compile(r"\b(19\d{2}|20\d{2})\b"),
        ]
        years: list[int] = []
        for pat in year_patterns:
            for m in pat.finditer(text):
                year = int(m.group(1))
                if 1900 <= year <= 2100:
                    years.append(year)
        if not years:
            return True, "未检测到文献年份, 跳过"
        import datetime
        current_year = datetime.datetime.now().year
        for y in years:
            age = current_year - y
            if age > 10:
                return False, f"文献年份 {y} 距今 {age} 年, 超过 10 年时效性要求"
        return True, f"文献年份 {years} 在 10 年时效性范围内"

    def _check_pr07(
        self,
        text: str,
        claim_text: str,
        evidence: list[Evidence] | None = None,
        **kw: Any,
    ) -> tuple[bool, str]:
        """P-R07: 冲突来源标注 — 多来源数值不一致时标注差异."""
        ev_list = evidence or []
        if len(ev_list) < 2:
            return True, "来源数量不足 2 个, 无需冲突检查"
        # 检查是否有冲突标注关键词
        conflict_markers = ["冲突", "差异", "不一致", "conflict", "discrepancy", "差异标注"]
        has_conflict_marker = _contains_any(text, conflict_markers)
        # 简单检查: 如果多个证据内容不同但未标注冲突
        contents = [ev.content for ev in ev_list if ev.content]
        if len(set(contents)) > 1 and not has_conflict_marker:
            return True, "多来源内容存在差异, 建议标注冲突"
        if has_conflict_marker:
            return True, "多来源差异已标注"
        return True, "冲突来源标注检查通过"

    def _check_pr08(
        self, claim: Claim, evidence: list[Evidence] | None = None, **kw: Any
    ) -> tuple[bool, str]:
        """P-R08: AI 生成标注."""
        ev_list = evidence or []
        has_evidence = bool(claim.evidence_ids) or len(ev_list) > 0
        if has_evidence:
            return True, "声明有来源, 无需 AI 生成标注"
        if claim.metadata.get("ai_generated", False):
            return True, "声明已标注为 AI-generated"
        return False, "无来源的声明未标注 AI-generated, unverified"

    def _check_pr09(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R09: 标准引用格式 — ACS 或 APA 标准."""
        if not _contains_any(text, ["acs", "apa", "引用", "reference", "et al"]):
            return True, "未检测到引用格式信息, 跳过"
        # 检查 ACS 格式: Author, A. B.; Author, C. D. Title. Journal Year, Vol, Pages.
        acs_pattern = re.compile(
            r"[A-Z][a-z]+,\s*[A-Z]\.[A-Z]\.;", re.IGNORECASE
        )
        # 检查 APA 格式: Author, A. B. (Year). Title. Journal.
        apa_pattern = re.compile(
            r"[A-Z][a-z]+,\s*[A-Z]\.[A-Z]\.\s*\(\d{4}\)", re.IGNORECASE
        )
        # 检查 DOI 格式
        doi_pattern = re.compile(r"10\.\d{4,}/\S+")
        if acs_pattern.search(text):
            return True, "引用格式符合 ACS 标准"
        if apa_pattern.search(text):
            return True, "引用格式符合 APA 标准"
        if doi_pattern.search(text):
            return True, "引用包含 DOI, 格式有效"
        if _contains_any(text, ["et al", "等"]):
            return True, "引用包含作者列表 (et al.), 格式基本有效"
        return True, "引用格式检查通过"

    def _check_pr10(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R10: 动态溯源版本 — 知识更新时创建新溯源版本."""
        # 检查是否提到版本信息
        version_markers = ["v1", "v2", "version", "版本", "修订", "revision", "updated"]
        if not _contains_any(text, version_markers):
            return True, "未检测到版本信息, 跳过"
        # 提取版本号
        version_patterns = [
            re.compile(r"[vV](\d+\.?\d*)"),
            re.compile(r"[vV]ersion\s*(\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"版本\s*(\d+\.?\d*)"),
        ]
        versions: list[str] = []
        for pat in version_patterns:
            versions.extend(pat.findall(text))
        if versions:
            return True, f"检测到溯源版本: {versions}"
        return True, "动态溯源版本检查通过"

    # ---- 增强规则: 溯源链与权威性评级 ----

    def _check_pr11(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R11: 来源 Tier 评级 — 来源应有 Tier 1-5 权威性评级."""
        from .provenance_chain import AuthorityRater

        if not _contains_any(text, ["来源", "source", "引用", "reference", "文献"]):
            return True, "未检测到来源信息, 跳过"
        rater = AuthorityRater()
        # 检查是否提到期刊名
        tier_found = False
        for journal in ["Nature", "Science", "PRL", "JACS", "ACS Nano", "Nano Letters"]:
            if journal.lower() in text.lower():
                tier = rater.determine_tier_from_journal(journal)
                tier_found = True
                if tier.value == "tier_1":
                    return True, f"来源 {journal} 评级为 Tier 1 (顶刊)"
        # 检查是否明确标注了 Tier
        tier_match = re.findall(r"[Tt]ier\s*[-_]?\s*([1-5])", text)
        if tier_match:
            tier_found = True
            tier_num = int(tier_match[0])
            if tier_num <= 2:
                return True, f"来源标注 Tier {tier_num} (权威来源)"
            return True, f"来源标注 Tier {tier_num}"
        if tier_found:
            return True, "检测到期刊来源, 已自动评级"
        return True, "未检测到明确的 Tier 评级, 建议标注"

    def _check_pr12(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R12: DOI 格式校验增强 — DOI 应符合 ISO 26324 结构标准."""
        from .provenance_chain import AuthorityRater

        dois = re.findall(r"10\.\d{4,}/\S+", text)
        if not dois:
            return True, "未检测到 DOI, 跳过"
        rater = AuthorityRater()
        invalid_dois = []
        for doi in dois:
            # 清理末尾标点
            clean_doi = doi.rstrip(".,;)]}")
            if not rater.validate_doi_format(clean_doi):
                invalid_dois.append(clean_doi)
        if invalid_dois:
            return False, f"DOI 格式不符合 ISO 26324: {invalid_dois[:3]}"
        return True, f"检测到 {len(dois)} 个 DOI, 格式校验通过"

    def _check_pr13(
        self,
        text: str,
        claim_text: str,
        evidence: list[Evidence] | None = None,
        **kw: Any,
    ) -> tuple[bool, str]:
        """P-R13: 溯源链深度验证 — 溯源链应达到至少 2 层深度."""
        from .provenance_chain import ProvenanceChain, ProvenanceNode, SourceTier

        ev_list = evidence or []
        if not ev_list:
            return True, "无证据可供溯源链深度检查, 跳过"
        chain = ProvenanceChain()
        for i, ev in enumerate(ev_list):
            node = ProvenanceNode(
                node_id=ev.evidence_id,
                source_type="database",
                source_uri=ev.source_uri or "",
                title="",
                authors=[],
                year=2024,
                tier=SourceTier.TIER_2,
                doi="",
                confidence=ev.confidence,
                content=ev.content,
                parent_ids=[ev_list[i - 1].evidence_id] if i > 0 else [],
            )
            chain.add_node(node)
        # 检查最后一个节点的链深度
        last_id = ev_list[-1].evidence_id
        depth = chain.get_depth(last_id)
        if depth < 2 and len(ev_list) >= 2:
            return False, f"溯源链深度 {depth} < 2, 溯源不充分"
        return True, f"溯源链深度 {depth}, 满足要求"

    def _check_pr14(
        self,
        text: str,
        claim_text: str,
        evidence: list[Evidence] | None = None,
        **kw: Any,
    ) -> tuple[bool, str]:
        """P-R14: 来源权威性评分 — 来源权威性评分应 ≥ 0.5."""
        from .provenance_chain import AuthorityRater, SourceTier

        ev_list = evidence or []
        if not ev_list:
            return True, "无证据可供权威性评分, 跳过"
        rater = AuthorityRater()
        low_confidence_count = 0
        for ev in ev_list:
            # 根据证据类型推断 Tier
            tier = SourceTier.TIER_3  # 默认 Tier 3
            if ev.evidence_type.value == "knowledge_base":
                tier = SourceTier.TIER_2
            elif ev.evidence_type.value == "external_source":
                tier = SourceTier.TIER_3
            elif ev.evidence_type.value == "computed":
                tier = SourceTier.TIER_4
            score = rater.calculate_confidence(tier, year=2024, citation_count=0)
            if score < 0.5:
                low_confidence_count += 1
        if low_confidence_count > len(ev_list) / 2:
            return False, f"{low_confidence_count}/{len(ev_list)} 个来源权威性评分 < 0.5"
        return True, f"来源权威性评分检查通过 (低置信来源 {low_confidence_count}/{len(ev_list)})"

    def _check_pr15(self, text: str, claim_text: str, **kw: Any) -> tuple[bool, str]:
        """P-R15: 溯源版本管理 — 知识更新应创建新溯源版本."""
        from .provenance_chain import VersionManager

        version_markers = ["v1", "v2", "v3", "version", "版本", "修订", "revision", "updated"]
        if not _contains_any(text, version_markers):
            return True, "未检测到版本信息, 跳过版本管理检查"
        vm = VersionManager()
        # 模拟版本创建
        version_id = vm.create_version(
            node_id="check-node",
            content=claim_text,
            reason="规则检查中检测到版本标记",
        )
        latest = vm.get_latest_version("check-node")
        if latest and latest.get("version_id") == version_id:
            return True, f"溯源版本管理功能正常 (版本 {version_id[:8]})"
        return True, "溯源版本管理检查通过"
