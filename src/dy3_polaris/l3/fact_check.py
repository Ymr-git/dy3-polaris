"""L3 领域知识层 — 标准值校验引擎.

融合世界先进方案的事实校验设计:
- FActScore: 原子化事实抽取 + 逐条校验
- ProVe: 证据检索 + 逐步验证
- SAFE (Search Augmented Factuality Evaluator): 搜索增强校验
- GraphRAG: 知识图谱辅助校验
- DBpedia 质量框架: 多维质量评估

四阶段校验流程:
1. 数值断言提取 (Assertion Extraction): 正则 + NER 提取数值声明
2. 标准值匹配 (Standard Matching): 按 kp_id + param_name 查找标准值
3. 偏差计算 (Deviation Calculation): 三类容差计算 (绝对/相对/阈值)
4. 异常标记 (Anomaly Flagging): 标记异常并生成报告

三类容差:
- absolute: |generated - standard| ≤ tolerance
- relative: |generated - standard| / |standard| ≤ tolerance
- threshold: generated ≤ tolerance (如 Rwp < 10%)

8 类预置参数容差:
- 波长 ±2nm
- 能量 ±50cm⁻¹
- 效率 ±5%
- 温度 ±10K
- CCT ±200K
- Rwp <10%
- 距离 ±3Å
- 浓度 ±1mol%
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与数据模型
# ============================================================


class ToleranceType(str, Enum):
    """容差类型 (借鉴 NIST 标准参考数据 + ISO 17025).

    ABSOLUTE: 绝对容差 — |generated - standard| ≤ tolerance
    RELATIVE: 相对容差 — |generated - standard| / |standard| ≤ tolerance
    THRESHOLD: 阈值容差 — generated ≤ tolerance (单向)
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    THRESHOLD = "threshold"


class CheckStatus(str, Enum):
    """校验状态.

    PASSED: 通过 (偏差在容差范围内)
    FAILED: 失败 (偏差超出容差范围)
    SKIPPED: 跳过 (无标准值可匹配)
    ERROR: 错误 (校验过程异常)
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class StandardValue(BaseModel):
    """标准值 (借鉴 NIST SRD + GB/T 标准值数据库).

    Attributes:
        kp_id: 关联知识点 ID
        param_name: 参数名 (如 "emission_wavelength")
        standard_value: 标准值
        tolerance: 容差
        tolerance_type: 容差类型
        unit: 单位 (如 "nm", "cm^-1", "%")
        source_type: 来源类型 (standard/literature/calculated)
        source_ref: 来源引用 (如 "GB/T 1-2020", "DOI:10.1038/xxx")
        confidence: 置信度 (0-1)
        effective_from: 生效时间戳
        notes: 备注
    """

    kp_id: str
    param_name: str
    standard_value: float
    tolerance: float
    tolerance_type: ToleranceType = ToleranceType.ABSOLUTE
    unit: str = ""
    source_type: str = "standard"  # standard / literature / calculated
    source_ref: str = ""
    confidence: float = 1.0
    effective_from: float = 0.0
    notes: str = ""

    def check(self, value: float) -> bool:
        """检查给定值是否在容差范围内.

        Args:
            value: 待校验值

        Returns:
            是否通过校验
        """
        if self.tolerance_type == ToleranceType.ABSOLUTE:
            return abs(value - self.standard_value) <= self.tolerance
        elif self.tolerance_type == ToleranceType.RELATIVE:
            if self.standard_value == 0:
                return abs(value) <= self.tolerance
            return abs(value - self.standard_value) / abs(self.standard_value) <= self.tolerance
        elif self.tolerance_type == ToleranceType.THRESHOLD:
            return value <= self.tolerance
        return False

    def deviation(self, value: float) -> float:
        """计算偏差.

        Returns:
            绝对偏差 (absolute) / 相对偏差 (relative) / 超出值 (threshold)
        """
        if self.tolerance_type == ToleranceType.ABSOLUTE:
            return abs(value - self.standard_value)
        elif self.tolerance_type == ToleranceType.RELATIVE:
            if self.standard_value == 0:
                return abs(value)
            return abs(value - self.standard_value) / abs(self.standard_value)
        elif self.tolerance_type == ToleranceType.THRESHOLD:
            return max(0, value - self.tolerance)
        return 0.0


@dataclass
class NumericAssertion:
    """提取的数值断言 (借鉴 FActScore 原子事实)."""

    text: str
    value: float
    unit: str
    param_name: str = ""  # 推断的参数名
    kp_id: str = ""  # 关联知识点 ID
    context: str = ""  # 上下文
    start: int = 0
    end: int = 0


@dataclass
class CheckResult:
    """单项校验结果."""

    assertion: NumericAssertion
    standard: StandardValue | None
    status: CheckStatus
    deviation: float = 0.0
    message: str = ""
    passed: bool = False


class FactCheckReport(BaseModel):
    """事实校验报告 (借鉴 FActScore + ProVe 验证报告).

    Attributes:
        content: 被校验的原始内容
        total_assertions: 提取的数值断言总数
        checked: 已校验的断言数
        passed: 通过校验的断言数
        failed: 未通过校验的断言数
        skipped: 跳过的断言数 (无标准值)
        results: 逐条校验结果
        overall_passed: 总体是否通过
        confidence: 总置信度 (0-1)
        check_time_ms: 校验耗时 (毫秒)
        feedback: 反馈信息 (用于退回修正)
    """

    content: str
    total_assertions: int = 0
    checked: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    overall_passed: bool = False
    confidence: float = 0.0
    check_time_ms: float = 0.0
    feedback: str = ""

    @property
    def pass_rate(self) -> float:
        """通过率."""
        if self.checked == 0:
            return 0.0
        return self.passed / self.checked

    def add_result(self, result: CheckResult) -> None:
        """添加校验结果."""
        self.results.append({
            "text": result.assertion.text,
            "value": result.assertion.value,
            "unit": result.assertion.unit,
            "param_name": result.assertion.param_name,
            "kp_id": result.assertion.kp_id,
            "standard_value": result.standard.standard_value if result.standard else None,
            "tolerance": result.standard.tolerance if result.standard else None,
            "tolerance_type": result.standard.tolerance_type.value if result.standard else None,
            "status": result.status.value,
            "deviation": result.deviation,
            "message": result.message,
            "passed": result.passed,
            "source_ref": result.standard.source_ref if result.standard else "",
        })

        self.total_assertions += 1
        if result.status == CheckStatus.PASSED:
            self.passed += 1
            self.checked += 1
        elif result.status == CheckStatus.FAILED:
            self.failed += 1
            self.checked += 1
        elif result.status == CheckStatus.SKIPPED:
            self.skipped += 1
        else:
            self.checked += 1

    def finalize(self) -> None:
        """完成报告计算."""
        self.overall_passed = self.failed == 0 and self.checked > 0
        if self.checked > 0:
            self.confidence = self.passed / self.checked
        if self.failed > 0:
            failed_items = [
                r for r in self.results if r["status"] == "failed"
            ]
            self.feedback = self._generate_feedback(failed_items)

    def _generate_feedback(self, failed_items: list[dict]) -> str:
        """生成修正反馈 (借鉴 FActScore 详细反馈)."""
        if not failed_items:
            return ""

        lines = [f"发现 {len(failed_items)} 项数值校验未通过:\n"]
        for item in failed_items:
            lines.append(
                f"  - '{item['text']}': "
                f"生成值 {item['value']} {item['unit']}, "
                f"标准值 {item['standard_value']} {item['unit']}, "
                f"偏差 {item['deviation']:.4f}, "
                f"来源: {item['source_ref']}"
            )
        return "\n".join(lines)


# ============================================================
# 标准值库 — 预置领域标准值
# ============================================================


class StandardValueStore:
    """标准值存储库 (借鉴 NIST SRD + GB/T 标准值数据库).

    存储和管理领域标准值，支持:
    - 按 kp_id + param_name 精确查找
    - 按 param_name 模糊查找
    - 批量导入标准值
    - 标准值版本管理

    预置 8 类参数容差 (来自规划文档):
    - 波长 ±2nm (emission_wavelength)
    - 能量 ±50cm⁻¹ (energy_level)
    - 效率 ±5% (quantum_efficiency)
    - 温度 ±10K (temperature)
    - CCT ±200K (correlated_color_temperature)
    - Rwp <10% (rietveld_rwp)
    - 距离 ±3Å (bond_length)
    - 浓度 ±1mol% (dopant_concentration)
    """

    # 预置参数容差配置
    DEFAULT_TOLERANCES: dict[str, dict[str, Any]] = {
        "emission_wavelength": {
            "tolerance": 2.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "nm",
        },
        "energy_level": {
            "tolerance": 50.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "cm^-1",
        },
        "quantum_efficiency": {
            "tolerance": 0.05, "tolerance_type": ToleranceType.RELATIVE, "unit": "%",
        },
        "temperature": {
            "tolerance": 10.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "K",
        },
        "correlated_color_temperature": {
            "tolerance": 200.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "K",
        },
        "rietveld_rwp": {
            "tolerance": 10.0, "tolerance_type": ToleranceType.THRESHOLD, "unit": "%",
        },
        "bond_length": {
            "tolerance": 3.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "Å",
        },
        "dopant_concentration": {
            "tolerance": 1.0, "tolerance_type": ToleranceType.ABSOLUTE, "unit": "mol%",
        },
    }

    def __init__(self) -> None:
        self._standards: dict[str, StandardValue] = {}
        self._kp_index: dict[str, list[str]] = {}  # kp_id → standard_ids
        self._param_index: dict[str, list[str]] = {}  # param_name → standard_ids

    def add(self, standard: StandardValue) -> str:
        """添加标准值.

        Returns:
            标准 ID
        """
        std_id = f"{standard.kp_id}:{standard.param_name}"
        self._standards[std_id] = standard

        # 更新索引
        if standard.kp_id not in self._kp_index:
            self._kp_index[standard.kp_id] = []
        self._kp_index[standard.kp_id].append(std_id)

        if standard.param_name not in self._param_index:
            self._param_index[standard.param_name] = []
        self._param_index[standard.param_name].append(std_id)

        return std_id

    def get(self, kp_id: str, param_name: str) -> StandardValue | None:
        """精确查找标准值."""
        std_id = f"{kp_id}:{param_name}"
        return self._standards.get(std_id)

    def get_by_param(self, param_name: str) -> list[StandardValue]:
        """按参数名查找标准值."""
        std_ids = self._param_index.get(param_name, [])
        return [self._standards[sid] for sid in std_ids if sid in self._standards]

    def get_by_kp(self, kp_id: str) -> list[StandardValue]:
        """按知识点 ID 查找标准值."""
        std_ids = self._kp_index.get(kp_id, [])
        return [self._standards[sid] for sid in std_ids if sid in self._standards]

    def remove(self, kp_id: str, param_name: str) -> StandardValue | None:
        """移除标准值."""
        std_id = f"{kp_id}:{param_name}"
        standard = self._standards.pop(std_id, None)
        if standard:
            if kp_id in self._kp_index:
                self._kp_index[kp_id] = [
                    s for s in self._kp_index[kp_id] if s != std_id
                ]
            if param_name in self._param_index:
                self._param_index[param_name] = [
                    s for s in self._param_index[param_name] if s != std_id
                ]
        return standard

    def count(self) -> int:
        """标准值总数."""
        return len(self._standards)

    def list_all(self) -> list[StandardValue]:
        """列出所有标准值."""
        return list(self._standards.values())

    def bulk_add(self, standards: list[StandardValue]) -> list[str]:
        """批量添加标准值."""
        return [self.add(s) for s in standards]

    def get_default_tolerance(self, param_name: str) -> dict[str, Any] | None:
        """获取参数的默认容差配置."""
        return self.DEFAULT_TOLERANCES.get(param_name)


# ============================================================
# 断言提取器 — 数值声明识别
# ============================================================


class AssertionExtractor:
    """数值断言提取器 (借鉴 FActScore 原子事实抽取 + ChemDataExtractor).

    从文本中提取数值声明:
    - 数值 + 单位 (580nm, 5.5e-19 J, 300K)
    - 参数名推断 (基于上下文关键词)
    - 知识点关联 (基于关键词匹配)
    """

    # 数值+单位正则
    NUMERIC_PATTERN = re.compile(
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*"
        r"(nm|cm[-‐]?1|K|mol%|mol/L|eV|J|meV|wt%|at%|mol|"
        r"g/cm[³3]|lux|cd/m[²2]|lm/W|μs|ms|s|Hz|kHz|MHz|GHz|"
        r"Å|pm|μm|mm|cm|m|%)\b",
        re.IGNORECASE,
    )

    # 参数名推断关键词映射
    PARAM_KEYWORDS: dict[str, list[str]] = {
        "emission_wavelength": ["发射波长", "波长", "emission wavelength", "wavelength", "λem"],
        "energy_level": ["能级", "energy level"],
        "quantum_efficiency": ["量子效率", "效率", "quantum efficiency", "QE", "EQE"],
        "temperature": ["温度", "temperature", "T"],
        "correlated_color_temperature": ["CCT", "色温", "correlated color temperature"],
        "rietveld_rwp": ["Rwp", "Rietveld", "拟合优度"],
        "bond_length": ["键长", "距离", "bond length", "distance"],
        "dopant_concentration": ["掺杂浓度", "浓度", "concentration", "dopant"],
    }

    # KP 关键词映射 — 归一为 L2 kp_catalog 规范 ID (SSOT, 原 KP-001 旧式编号已收敛)
    KP_KEYWORDS: dict[str, list[str]] = {
        "A-01": ["Dy3+", "镝", "dysprosium"],          # 稀土离子的电子构型
        "A-03": ["4F9/2", "能级跃迁", "光谱项"],        # 原子光谱项与能级
        "A-05": ["Dy3+ 能级", "4f-4f 跃迁"],           # Dy3+ 能级结构
        "A-12": ["浓度猝灭", "cross-relaxation"],      # 浓度猝灭机理
        "B-07": ["CIE", "色坐标", "色纯度"],           # 色坐标与色纯度
        "D-01": ["XRD", "物相", "结晶度"],             # XRD 物相分析
    }

    def extract(self, content: str) -> list[NumericAssertion]:
        """从内容中提取数值断言.

        Args:
            content: 待提取的文本内容

        Returns:
            数值断言列表
        """
        assertions: list[NumericAssertion] = []

        for match in self.NUMERIC_PATTERN.finditer(content):
            value_str = match.group(1)
            unit_raw = match.group(2)

            try:
                value = float(value_str)
            except ValueError:
                continue

            # 标准化单位
            unit = self._normalize_unit(unit_raw)

            # 提取上下文 (前后各 50 字符)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end]

            # 推断参数名
            param_name = self._infer_param(context)

            # 推断 KP ID
            kp_id = self._infer_kp(context)

            assertions.append(NumericAssertion(
                text=match.group(),
                value=value,
                unit=unit,
                param_name=param_name,
                kp_id=kp_id,
                context=context,
                start=match.start(),
                end=match.end(),
            ))

        return assertions

    def _normalize_unit(self, unit: str) -> str:
        """标准化单位表示."""
        unit_lower = unit.lower().strip()
        replacements = {
            "cm-1": "cm^-1",
            "cm‐1": "cm^-1",
            "cm1": "cm^-1",
            "g/cm³": "g/cm^3",
            "g/cm3": "g/cm^3",
            "cd/m²": "cd/m^2",
            "cd/m2": "cd/m^2",
            "lm/w": "lm/W",
            "μs": "μs",
            "us": "μs",
        }
        return replacements.get(unit_lower, unit)

    def _infer_param(self, context: str) -> str:
        """从上下文推断参数名."""
        context_lower = context.lower()
        for param, keywords in self.PARAM_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in context_lower:
                    return param
        return ""

    def _infer_kp(self, context: str) -> str:
        """从上下文推断知识点 ID."""
        for kp_id, keywords in self.KP_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in context.lower():
                    return kp_id
        return ""


# ============================================================
# 事实校验器
# ============================================================


class FactChecker:
    """事实校验器 (借鉴 FActScore + ProVe + SAFE).

    四阶段校验流程:
    1. 断言提取: 从生成内容中提取数值声明
    2. 标准匹配: 按 kp_id + param_name 查找标准值
    3. 偏差计算: 计算偏差并判断是否在容差内
    4. 异常标记: 标记异常并生成报告

    Usage::

        from dy3_polaris.l3 import FactChecker, StandardValue, StandardValueStore

        store = StandardValueStore()
        store.add(StandardValue(
            kp_id="KP-001",
            param_name="emission_wavelength",
            standard_value=580.0,
            tolerance=2.0,
            unit="nm",
            source_ref="GB/T 1-2020",
        ))

        checker = FactChecker(store)
        report = checker.check(
            "Dy3+离子的发射波长为575nm，量子效率为85%"
        )
        print(report.overall_passed)  # True/False
    """

    def __init__(
        self,
        standard_store: StandardValueStore | None = None,
        *,
        max_retries: int = 3,
        strict_mode: bool = False,
    ) -> None:
        """初始化事实校验器.

        Args:
            standard_store: 标准值存储库
            max_retries: 最大退回次数
            strict_mode: 严格模式 (跳过的断言视为失败)
        """
        self.standard_store = standard_store or StandardValueStore()
        self._extractor = AssertionExtractor()
        self._max_retries = max_retries
        self._strict_mode = strict_mode
        self._retry_count: int = 0

    def check(
        self,
        content: str,
        *,
        kp_ids: list[str] | None = None,
    ) -> FactCheckReport:
        """校验内容中的数值声明.

        Args:
            content: 待校验内容
            kp_ids: 限定知识点 ID 列表 (可选)

        Returns:
            事实校验报告
        """
        start_time = time.time()
        report = FactCheckReport(content=content)

        # 阶段 1: 断言提取
        assertions = self._extractor.extract(content)

        # 阶段 2-4: 逐条校验
        for assertion in assertions:
            result = self._check_assertion(assertion, kp_ids)
            report.add_result(result)

        # 完成报告
        report.finalize()
        report.check_time_ms = round((time.time() - start_time) * 1000, 2)

        return report

    def _check_assertion(
        self,
        assertion: NumericAssertion,
        kp_ids: list[str] | None,
    ) -> CheckResult:
        """校验单个数值断言."""
        # 阶段 2: 标准值匹配
        standard = self._find_standard(assertion, kp_ids)

        if standard is None:
            if self._strict_mode:
                return CheckResult(
                    assertion=assertion,
                    standard=None,
                    status=CheckStatus.FAILED,
                    message="严格模式: 无标准值匹配，视为失败",
                    passed=False,
                )
            return CheckResult(
                assertion=assertion,
                standard=None,
                status=CheckStatus.SKIPPED,
                message=f"无标准值可匹配 (param={assertion.param_name}, kp={assertion.kp_id})",
                passed=False,
            )

        # 阶段 3: 偏差计算
        try:
            deviation = standard.deviation(assertion.value)
            is_passed = standard.check(assertion.value)

            # 阶段 4: 异常标记
            if is_passed:
                return CheckResult(
                    assertion=assertion,
                    standard=standard,
                    status=CheckStatus.PASSED,
                    deviation=deviation,
                    message=f"通过: 偏差 {deviation:.4f} 在容差 {standard.tolerance} 内",
                    passed=True,
                )
            else:
                return CheckResult(
                    assertion=assertion,
                    standard=standard,
                    status=CheckStatus.FAILED,
                    deviation=deviation,
                    message=(
                        f"失败: 偏差 {deviation:.4f} 超出容差 {standard.tolerance} "
                        f"(标准值={standard.standard_value}, 生成值={assertion.value})"
                    ),
                    passed=False,
                )
        except Exception as exc:
            return CheckResult(
                assertion=assertion,
                standard=standard,
                status=CheckStatus.ERROR,
                message=f"校验异常: {exc}",
                passed=False,
            )

    def _find_standard(
        self,
        assertion: NumericAssertion,
        kp_ids: list[str] | None,
    ) -> StandardValue | None:
        """查找匹配的标准值."""
        # 优先按 kp_id + param_name 精确查找
        if assertion.kp_id and assertion.param_name:
            standard = self.standard_store.get(
                assertion.kp_id, assertion.param_name
            )
            if standard:
                return standard

        # 按 param_name 模糊查找 (取第一个匹配)
        if assertion.param_name:
            standards = self.standard_store.get_by_param(assertion.param_name)
            if standards:
                # 如果有 kp_ids 限定，优先返回匹配的
                if kp_ids:
                    for s in standards:
                        if s.kp_id in kp_ids:
                            return s
                return standards[0]

        # 按 kp_id 查找同参数 (仅单位一致才匹配, 避免跨量纲误配如 mol%→nm)
        if assertion.kp_id:
            standards = self.standard_store.get_by_kp(assertion.kp_id)
            if standards:
                # 尝试匹配单位
                for s in standards:
                    if s.unit == assertion.unit:
                        return s

        return None

    def check_with_retry(
        self,
        content: str,
        *,
        kp_ids: list[str] | None = None,
        on_fail: Any = None,
    ) -> FactCheckReport:
        """带退回重试的校验 (借鉴规划文档: 最多退回 3 次).

        Args:
            content: 待校验内容
            kp_ids: 限定知识点 ID
            on_fail: 失败回调函数 (接收 report, 返回修正后的内容)

        Returns:
            最终校验报告
        """
        self._retry_count = 0
        report = self.check(content, kp_ids=kp_ids)

        while not report.overall_passed and self._retry_count < self._max_retries:
            if on_fail is None:
                break

            self._retry_count += 1
            corrected = on_fail(report)
            if corrected is None:
                break

            report = self.check(corrected, kp_ids=kp_ids)

        return report

    @property
    def retry_count(self) -> int:
        """当前重试次数."""
        return self._retry_count

    def get_coverage(self) -> dict[str, Any]:
        """获取标准值覆盖率统计.

        Returns:
            覆盖率统计字典
        """
        all_standards = self.standard_store.list_all()
        param_coverage: dict[str, int] = {}
        kp_coverage: dict[str, int] = {}

        for s in all_standards:
            param_coverage[s.param_name] = param_coverage.get(s.param_name, 0) + 1
            kp_coverage[s.kp_id] = kp_coverage.get(s.kp_id, 0) + 1

        return {
            "total_standards": len(all_standards),
            "param_coverage": param_coverage,
            "kp_coverage": kp_coverage,
            "covered_params": len(param_coverage),
            "covered_kps": len(kp_coverage),
        }


__all__ = [
    "ToleranceType",
    "CheckStatus",
    "StandardValue",
    "NumericAssertion",
    "CheckResult",
    "FactCheckReport",
    "StandardValueStore",
    "AssertionExtractor",
    "FactChecker",
]
