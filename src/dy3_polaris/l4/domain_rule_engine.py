"""L4 决策引擎层 — 领域适配验证规则引擎.

借鉴世界先进方案:
- DomainShield (2025): 领域特定验证规则框架
  - 可插拔规则架构: 每个领域注册自己的验证规则
  - 规则优先级: 高优先级规则先执行，低优先级规则补充
  - 规则冲突消解: 当多个规则给出矛盾结论时的仲裁机制
- KGFact (2025): 知识图谱事实验证
  - 实体-属性-值三元组校验
  - 数值范围检查
  - 关系一致性验证
- SpecChecker (2025): 规格合规检查
  - 领域规格库管理
  - 规格匹配与偏差量化

核心职责:
    1. 注册和管理领域特定验证规则
    2. 对执行结果执行领域规则校验
    3. 生成领域特定的异常报告和修正建议
    4. 支持规则的动态加载和优先级调整

适用领域 (可扩展):
    - 稀土发光材料: 发光波长、激发态、浓度淬灭等
    - 半导体器件: 电学参数、能带结构、温度特性等
    - 催化材料: 活性位点、反应路径、选择性等
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from .models import ExecutionResult, TaskType

logger = logging.getLogger(__name__)


# ============================================================
# 规则基类
# ============================================================


class DomainRule(ABC):
    """领域验证规则基类.

    所有领域规则继承此类，实现 check 方法。
    规则具有优先级 (priority) 和启用状态 (enabled)。
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        *,
        priority: int = 50,
        enabled: bool = True,
        severity: str = "warning",
    ) -> None:
        """初始化规则.

        Args:
            rule_id: 规则唯一标识
            name: 规则名称
            priority: 优先级 (0-100, 越高越优先)
            enabled: 是否启用
            severity: 默认严重级别 (info/warning/error/critical)
        """
        self.rule_id = rule_id
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.severity = severity

    @abstractmethod
    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """执行规则检查.

        Args:
            execution_result: 执行结果

        Returns:
            检查结果字典:
            - rule_id: 规则 ID
            - rule_name: 规则名称
            - passed: 是否通过
            - score: 规则评分 (0~1)
            - violations: 违规列表
            - details: 详细信息
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """序列化规则元数据."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "severity": self.severity,
        }


# ============================================================
# 稀土发光材料领域规则
# ============================================================


class WavelengthRangeRule(DomainRule):
    """发射波长范围校验规则.

    检查稀土离子的发射波长是否在合理范围内:
    - Dy3+: 470-490nm (蓝光) + 570-590nm (黄光)
    - Eu3+: 590-630nm (红光)
    - Tb3+: 540-560nm (绿光)
    - Ce3+: 400-500nm (蓝光)
    - Sm3+: 600-660nm (橙红)
    """

    ION_WAVELENGTH_RANGES = {
        "Dy3+": [(470, 490, "蓝光 ^4F9/2"), (570, 590, "黄光 ^4F9/2")],
        "Eu3+": [(590, 630, "红光 ^5D0")],
        "Tb3+": [(540, 560, "绿光 ^5D4")],
        "Ce3+": [(400, 500, "蓝光 5d-4f")],
        "Sm3+": [(600, 660, "橙红 ^4G5/2")],
        "Nd3+": [(1050, 1100, "近红外 ^4F3/2")],
        "Er3+": [(1500, 1600, "近红外 ^4I13/2")],
        "Yb3+": [(970, 1000, "近红外 ^2F5/2")],
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="rare_earth_wavelength_range",
            name="稀土离子发射波长范围校验",
            priority=90,
            severity="error",
        )

    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """检查波长范围."""
        violations: list[dict[str, Any]] = []
        checked_count = 0

        for tr in execution_result.get_results_by_type(TaskType.REASON):
            answers = tr.output.get("answers", [])
            for ans in answers:
                if not isinstance(ans, dict):
                    continue
                text = str(ans.get("text", "")) + str(ans.get("value", ""))
                checked_count += self._check_text(text, violations)

        passed = len(violations) == 0
        score = 1.0 if checked_count == 0 else max(0.0, 1.0 - len(violations) / max(1, checked_count))

        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "passed": passed,
            "score": round(score, 4),
            "violations": violations,
            "details": {"checked_claims": checked_count},
        }

    def _check_text(self, text: str, violations: list[dict[str, Any]]) -> int:
        """检查文本中的波长声明."""
        count = 0
        pattern = re.compile(
            r"(Dy3\+|Eu3\+|Tb3\+|Ce3\+|Sm3\+|Nd3\+|Er3\+|Yb3\+).*?"
            r"(\d+\.?\d*)\s*nm",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            ion = match.group(1)
            wavelength = float(match.group(2))
            ranges = self.ION_WAVELENGTH_RANGES.get(ion, [])
            if not ranges:
                continue
            count += 1
            in_range = any(lo <= wavelength <= hi for lo, hi, _ in ranges)
            if not in_range:
                expected = " 或 ".join(f"{lo}-{hi}nm ({desc})" for lo, hi, desc in ranges)
                violations.append({
                    "type": "wavelength_out_of_range",
                    "ion": ion,
                    "actual_wavelength": wavelength,
                    "expected_range": expected,
                    "raw_text": match.group(0),
                    "severity": self.severity,
                })
        return count


class ConcentrationQuenchingRule(DomainRule):
    """浓度淬灭校验规则.

    检查浓度淬灭相关的描述是否符合物理规律:
    - 稀土离子掺杂浓度过高时，发光强度应下降
    - 交叉弛豫是常见的淬灭机理
    - 临界浓度因离子和基质不同而异
    """

    # 典型临界浓度 (mol%)
    CRITICAL_CONCENTRATIONS = {
        "Dy3+": 5.0,
        "Eu3+": 8.0,
        "Tb3+": 7.0,
        "Ce3+": 3.0,
        "Sm3+": 5.0,
        "Nd3+": 2.0,
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="concentration_quenching",
            name="浓度淬灭物理规律校验",
            priority=80,
            severity="warning",
        )

    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """检查浓度淬灭描述."""
        violations: list[dict[str, Any]] = []
        checked_count = 0

        for tr in execution_result.task_results.values():
            output_text = str(tr.output.get("summary", "")) + str(tr.output.get("answers", ""))
            checked_count += self._check_concentration_logic(output_text, violations)

        passed = len(violations) == 0
        score = 1.0 if checked_count == 0 else max(0.0, 1.0 - len(violations) * 0.3)

        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "passed": passed,
            "score": round(score, 4),
            "violations": violations,
            "details": {"checked_claims": checked_count},
        }

    def _check_concentration_logic(self, text: str, violations: list[dict[str, Any]]) -> int:
        """检查浓度与发光强度的关系."""
        count = 0
        # 匹配 "X离子 浓度 Y% 发光强度 Z"
        pattern = re.compile(
            r"(Dy3\+|Eu3\+|Tb3\+|Ce3\+|Sm3\+|Nd3\+).*?"
            r"浓度.*?(\d+\.?\d*)\s*(?:mol%|%|at%).*?"
            r"(?:发光强度|发光效率|荧光强度|量子效率).*?"
            r"(提高|增强|上升|增加|变大|更高|提升|增大)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            ion = match.group(1)
            concentration = float(match.group(2))
            critical = self.CRITICAL_CONCENTRATIONS.get(ion, 5.0)
            count += 1
            if concentration > critical * 1.5:
                violations.append({
                    "type": "concentration_quenching_violation",
                    "ion": ion,
                    "concentration": concentration,
                    "critical_concentration": critical,
                    "issue": f"浓度 {concentration}% 远超临界浓度 {critical}%，但声称发光强度提高，违反浓度淬灭规律",
                    "severity": self.severity,
                })
        return count


class EnergyTransferRule(DomainRule):
    """能量传递校验规则.

    检查能量传递描述的物理合理性:
    - 敏化剂发射光谱应与激活剂吸收光谱重叠
    - 能量传递效率应合理 (0-100%)
    - 能量传递方向应符合能级关系
    """

    # 常见敏化剂-激活剂对的能级关系
    KNOWN_PAIRS = {
        ("Ce3+", "Dy3+"): "Ce3+ 5d -> Dy3+ ^4F9/2",
        ("Ce3+", "Tb3+"): "Ce3+ 5d -> Tb3+ ^5D4",
        ("Eu2+", "Mn2+"): "Eu2+ 4f65d1 -> Mn2+ 4T1",
        ("Bi3+", "Eu3+"): "Bi3+ ^3P1 -> Eu3+ ^5D0",
        ("Tb3+", "Eu3+"): "Tb3+ ^5D4 -> Eu3+ ^5D0",
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="energy_transfer",
            name="能量传递物理合理性校验",
            priority=75,
            severity="warning",
        )

    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """检查能量传递描述."""
        violations: list[dict[str, Any]] = []
        checked_count = 0

        for tr in execution_result.task_results.values():
            text = str(tr.output.get("summary", "")) + str(tr.output.get("answers", ""))
            checked_count += self._check_energy_transfer(text, violations)

        passed = len(violations) == 0
        score = 1.0 if checked_count == 0 else max(0.0, 1.0 - len(violations) * 0.25)

        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "passed": passed,
            "score": round(score, 4),
            "violations": violations,
            "details": {"checked_claims": checked_count},
        }

    def _check_energy_transfer(self, text: str, violations: list[dict[str, Any]]) -> int:
        """检查能量传递描述."""
        count = 0
        # 匹配 "X 到 Y 的能量传递" 或 "X 向 Y 传递能量"
        pattern = re.compile(
            r"(Ce3\+|Eu2\+|Eu3\+|Bi3\+|Tb3\+|Dy3\+|Mn2\+|Yb3\+|Er3\+).*?"
            r"(?:到|向|至).*?"
            r"(Ce3\+|Eu2\+|Eu3\+|Bi3\+|Tb3\+|Dy3\+|Mn2\+|Yb3\+|Er3\+).*?"
            r"(?:能量传递|能量转移|Energy Transfer)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            sensitizer = match.group(1)
            activator = match.group(2)
            if sensitizer == activator:
                continue
            count += 1
            pair = (sensitizer, activator)
            if pair not in self.KNOWN_PAIRS:
                violations.append({
                    "type": "unknown_energy_transfer_pair",
                    "sensitizer": sensitizer,
                    "activator": activator,
                    "issue": f"非标准敏化剂-激活剂对 {sensitizer} -> {activator}，请验证能级匹配性",
                    "severity": "info",
                })
        return count


class CrystalFieldRule(DomainRule):
    """晶体场效应校验规则.

    检查晶体场相关的描述:
    - 不同基质中同一离子的发射波长应有差异
    - 晶体场分裂能应在合理范围
    - 对称性影响光谱特性
    """

    # 常见基质中 Dy3+ 的 ^4F9/2 发射波长 (nm)
    DY3_HOST_WAVELENGTHS = {
        "YAG": 575,
        "Y2O3": 573,
        "La2O3": 576,
        "Gd2O3": 575,
        "Lu2O3": 574,
        "YVO4": 575,
        "YPO4": 576,
        "BaYF5": 575,
        "NaYF4": 574,
        "CaF2": 576,
    }

    def __init__(self) -> None:
        super().__init__(
            rule_id="crystal_field_effect",
            name="晶体场效应一致性校验",
            priority=70,
            severity="info",
        )

    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """检查晶体场效应."""
        violations: list[dict[str, Any]] = []
        checked_count = 0

        for tr in execution_result.task_results.values():
            text = str(tr.output.get("summary", "")) + str(tr.output.get("answers", ""))
            checked_count += self._check_host_wavelength(text, violations)

        passed = len(violations) == 0
        score = 1.0 if checked_count == 0 else max(0.0, 1.0 - len(violations) * 0.2)

        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "passed": passed,
            "score": round(score, 4),
            "violations": violations,
            "details": {"checked_claims": checked_count},
        }

    def _check_host_wavelength(self, text: str, violations: list[dict[str, Any]]) -> int:
        """检查基质-波长一致性."""
        count = 0
        for host, expected_wl in self.DY3_HOST_WAVELENGTHS.items():
            pattern = re.compile(
                rf"{re.escape(host)}.*?Dy3\+.*?(\d+\.?\d*)\s*nm",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                actual_wl = float(match.group(1))
                count += 1
                deviation = abs(actual_wl - expected_wl)
                if deviation > 15:
                    violations.append({
                        "type": "wavelength_host_mismatch",
                        "host": host,
                        "actual_wavelength": actual_wl,
                        "expected_wavelength": expected_wl,
                        "deviation": round(deviation, 2),
                        "issue": f"{host} 中 Dy3+ 发射波长通常约 {expected_wl}nm，实际值 {actual_wl}nm 偏差较大",
                        "severity": "info",
                    })
        return count


# ============================================================
# 通用数值范围规则
# ============================================================


class NumericRangeRule(DomainRule):
    """通用数值范围校验规则.

    可配置参数的数值范围检查，适用于:
    - 温度范围
    - 量子效率范围
    - 衰减寿命范围
    - 颗粒尺寸范围
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        param_pattern: str,
        min_value: float,
        max_value: float,
        *,
        unit: str = "",
        priority: int = 60,
        severity: str = "warning",
    ) -> None:
        super().__init__(rule_id, name, priority=priority, severity=severity)
        self._param_pattern = re.compile(param_pattern, re.IGNORECASE)
        self._min_value = min_value
        self._max_value = max_value
        self._unit = unit

    def check(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """检查数值范围."""
        violations: list[dict[str, Any]] = []
        checked_count = 0

        for tr in execution_result.task_results.values():
            text = str(tr.output.get("summary", "")) + str(tr.output.get("answers", ""))
            for match in self._param_pattern.finditer(text):
                value = float(match.group(1))
                checked_count += 1
                if value < self._min_value or value > self._max_value:
                    violations.append({
                        "type": "numeric_out_of_range",
                        "actual_value": value,
                        "min": self._min_value,
                        "max": self._max_value,
                        "unit": self._unit,
                        "raw_text": match.group(0),
                        "severity": self.severity,
                    })

        passed = len(violations) == 0
        score = 1.0 if checked_count == 0 else max(0.0, 1.0 - len(violations) / max(1, checked_count))

        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "passed": passed,
            "score": round(score, 4),
            "violations": violations,
            "details": {"checked_claims": checked_count},
        }


# ============================================================
# 领域规则引擎
# ============================================================


class DomainRuleEngine:
    """领域适配验证规则引擎.

    管理多个领域验证规则，按优先级执行，
    汇总所有规则的检查结果。

    Usage::

        engine = DomainRuleEngine()
        engine.register_rule(WavelengthRangeRule())
        engine.register_rule(ConcentrationQuenchingRule())
        result = engine.evaluate(execution_result)
        # result.score -> 领域规则综合评分
        # result.violations -> 所有违规列表
    """

    # 默认规则集
    DEFAULT_RULES: list[DomainRule] = []

    def __init__(self, *, auto_load_defaults: bool = True) -> None:
        """初始化规则引擎.

        Args:
            auto_load_defaults: 是否自动加载默认规则集
        """
        self._rules: dict[str, DomainRule] = {}

        if auto_load_defaults:
            self._load_default_rules()

        logger.info(
            "DomainRuleEngine 初始化 (已加载 %d 条规则)",
            len(self._rules),
        )

    def _load_default_rules(self) -> None:
        """加载默认领域规则集."""
        defaults = [
            WavelengthRangeRule(),
            ConcentrationQuenchingRule(),
            EnergyTransferRule(),
            CrystalFieldRule(),
            NumericRangeRule(
                rule_id="temperature_range",
                name="温度范围合理性校验",
                param_pattern=r"温度.*?(\d+\.?\d*)\s*(?:K|℃)",
                min_value=0,
                max_value=3000,
                unit="K",
                priority=65,
                severity="warning",
            ),
            NumericRangeRule(
                rule_id="quantum_efficiency_range",
                name="量子效率范围校验",
                param_pattern=r"量子效率.*?(\d+\.?\d*)\s*%",
                min_value=0,
                max_value=100,
                unit="%",
                priority=65,
                severity="warning",
            ),
            NumericRangeRule(
                rule_id="decay_lifetime_range",
                name="衰减寿命范围校验",
                param_pattern=r"衰减寿命.*?(\d+\.?\d*)\s*(?:μs|ms|ns|us)",
                min_value=0.001,
                max_value=10000,
                unit="μs",
                priority=60,
                severity="info",
            ),
        ]
        for rule in defaults:
            self.register_rule(rule)

    def register_rule(self, rule: DomainRule) -> None:
        """注册一条验证规则.

        Args:
            rule: 验证规则实例
        """
        if rule.rule_id in self._rules:
            logger.warning("规则 %s 已存在，将被覆盖", rule.rule_id)
        self._rules[rule.rule_id] = rule
        logger.debug("注册规则: %s (优先级: %d)", rule.rule_id, rule.priority)

    def unregister_rule(self, rule_id: str) -> bool:
        """注销一条验证规则.

        Args:
            rule_id: 规则 ID

        Returns:
            是否成功注销
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.debug("注销规则: %s", rule_id)
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """启用规则."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def get_rules(self) -> list[dict[str, Any]]:
        """获取所有规则元数据."""
        return [r.to_dict() for r in self._rules.values()]

    def evaluate(self, execution_result: ExecutionResult) -> dict[str, Any]:
        """执行所有启用的规则.

        Args:
            execution_result: 执行结果

        Returns:
            评估结果字典:
            - overall_score: 领域规则综合评分 (0~1)
            - total_rules: 总规则数
            - executed_rules: 已执行规则数
            - passed_rules: 通过规则数
            - failed_rules: 未通过规则数
            - rule_results: 各规则检查结果
            - all_violations: 所有违规列表
            - high_severity_violations: 高严重级别违规
        """
        # 按优先级排序 (降序)
        sorted_rules = sorted(
            self._rules.values(),
            key=lambda r: r.priority,
            reverse=True,
        )

        rule_results: list[dict[str, Any]] = []
        all_violations: list[dict[str, Any]] = []
        high_severity_violations: list[dict[str, Any]] = []

        passed_count = 0
        failed_count = 0
        scores: list[float] = []

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            try:
                result = rule.check(execution_result)
                rule_results.append(result)
                scores.append(result["score"])

                if result["passed"]:
                    passed_count += 1
                else:
                    failed_count += 1

                for violation in result.get("violations", []):
                    all_violations.append(violation)
                    if violation.get("severity") in ("error", "critical"):
                        high_severity_violations.append(violation)

            except Exception as exc:  # noqa: BLE001
                logger.exception("规则 %s 执行异常", rule.rule_id)
                rule_results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "passed": False,
                    "score": 0.5,
                    "violations": [],
                    "error": str(exc),
                })
                failed_count += 1
                scores.append(0.5)

        total_executed = passed_count + failed_count
        overall_score = sum(scores) / len(scores) if scores else 1.0

        # 高严重级别违规降低整体评分
        if high_severity_violations:
            penalty = min(0.3, len(high_severity_violations) * 0.1)
            overall_score = max(0.0, overall_score - penalty)

        return {
            "overall_score": round(overall_score, 4),
            "total_rules": len(self._rules),
            "executed_rules": total_executed,
            "passed_rules": passed_count,
            "failed_rules": failed_count,
            "rule_results": rule_results,
            "all_violations": all_violations,
            "high_severity_violations": high_severity_violations,
        }


__all__ = [
    "DomainRule",
    "DomainRuleEngine",
    "WavelengthRangeRule",
    "ConcentrationQuenchingRule",
    "EnergyTransferRule",
    "CrystalFieldRule",
    "NumericRangeRule",
]
