"""CC1 数值层 — 计算验证与单位转换模块.

本模块为 L3 NumericalLayer（数值校验层）提供底层计算支撑，聚焦
Dy3+ 发光材料领域的物理量换算、Judd-Ofelt 理论计算、CIE 色度学
计算以及误差与不确定度分析。模块仅依赖 Python 标准库
(``math``, ``statistics``, ``re``)，可独立使用，无第三方依赖。

主要组件
--------
- :class:`UnitConverter`        物理单位换算（nm ↔ cm⁻¹ ↔ eV ↔ kJ/mol）
- :class:`JuddOfeltCalculator`  Judd-Ofelt 理论相关计算
- :class:`CIECalculator`        CIE 1931 色度学计算
- :class:`ErrorAnalyzer`        误差传播与不确定度分析

设计参考
--------
- Judd-Ofelt 理论: B. R. Judd (1962), G. S. Ofelt (1962)
- CIE 1931 色度学: CIE 技术报告（2° 标准观察者）
- McCamy 色温近似公式: McCamy (1992)
- 误差传播: ISO/IEC Guide 98-3《测量不确定度表示指南》(GUM)
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable


# ============================================================
# 物理常量
# ============================================================


@dataclass(frozen=True)
class PhysicalConstants:
    """物理常量集合（SI 单位）.

    所有计算均基于此数据类提供的常量，便于在测试或高精度场景中
    替换为 CODATA 推荐值。

    Attributes:
        h:         普朗克常量 (J·s)
        c:         真空光速 (m/s)
        e:         元电荷 (C)
        N_A:       阿伏伽德罗常量 (mol⁻¹)
        m_e:       电子静止质量 (kg)
        epsilon_0: 真空电容率 (F/m)
    """

    h: float = 6.626e-34
    c: float = 3.0e8
    e: float = 1.602e-19
    N_A: float = 6.022e23
    m_e: float = 9.109e-31
    epsilon_0: float = 8.854e-12


#: 全局物理常量实例
CONST: PhysicalConstants = PhysicalConstants()


# ============================================================
# 异常
# ============================================================


class ComputationError(ValueError):
    """计算验证与单位转换过程中的通用异常."""


# ============================================================
# 辅助函数
# ============================================================


def _require_positive(value: float, name: str) -> float:
    """校验数值为严格正数并返回 ``float``，否则抛出 :class:`ComputationError`."""
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ComputationError(f"{name} 必须为数值, 收到 {value!r}") from exc
    if not math.isfinite(fval) or fval <= 0.0:
        raise ComputationError(f"{name} 必须为正有限数, 收到 {fval}")
    return fval


def _require_finite(value: float, name: str) -> float:
    """校验数值为有限数并返回 ``float``."""
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ComputationError(f"{name} 必须为数值, 收到 {value!r}") from exc
    if not math.isfinite(fval):
        raise ComputationError(f"{name} 必须为有限数, 收到 {fval}")
    return fval


def _quantile(sorted_data: list[float], q: float) -> float:
    """计算有序序列的分位数（线性插值法）.

    Args:
        sorted_data: 已升序排列的数值列表。
        q:           分位数位置，取值 [0, 1]。

    Returns:
        对应分位数值。
    """
    if not sorted_data:
        raise ComputationError("分位数计算需要非空数据集")
    if not 0.0 <= q <= 1.0:
        raise ComputationError(f"分位位置 q 必须在 [0, 1] 内, 收到 {q}")
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_data[lo]
    frac = pos - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


# ============================================================
# UnitConverter — 物理单位换算
# ============================================================


class UnitConverter:
    """物理单位换算器.

    提供 Dy3+ 发光材料常用物理量之间的换算：波长 (nm)、波数 (cm⁻¹)、
    光子能量 (eV)、摩尔能量 (kJ/mol)。所有换算基于 SI 物理常量。

    换算关系::

        波数 ν̃ (cm⁻¹) = 10⁷ / λ(nm)
        能量 E (eV)    = hc / (λ·e)        , λ 单位 m
        能量 E (kJ/mol)= hc·N_A / (λ·1000) , λ 单位 m

    使用示例::

        >>> uc = UnitConverter()
        >>> round(uc.nm_to_cm_inv(500), 1)
        20000.0
        >>> round(uc.nm_to_ev(575), 1)
        2.2
    """

    #: 支持的规范单位标识
    SUPPORTED_UNITS: tuple[str, ...] = ("nm", "cm-1", "ev", "kj/mol")

    #: 单位别名 → 规范单位（用于 ``validate_conversion`` 的宽松解析）
    _UNIT_ALIASES: dict[str, str] = {
        "nm": "nm",
        "nanometer": "nm",
        "nanometers": "nm",
        "纳米": "nm",
        "cm-1": "cm-1",
        "cm^-1": "cm-1",
        "cm⁻¹": "cm-1",
        "1/cm": "cm-1",
        "wavenumber": "cm-1",
        "波数": "cm-1",
        "ev": "ev",
        "electronvolt": "ev",
        "electronvolts": "ev",
        "电子伏特": "ev",
        "kj/mol": "kj/mol",
        "kj_per_mol": "kj/mol",
        "千焦每摩尔": "kj/mol",
        "kj·mol⁻¹": "kj/mol",
    }

    def __init__(self, constants: PhysicalConstants | None = None) -> None:
        self._c = constants or CONST

    # ------------------------------------------------------------------
    # 基础换算
    # ------------------------------------------------------------------

    def nm_to_cm_inv(self, nm: float) -> float:
        """波长 (nm) → 波数 (cm⁻¹).

        公式::

            ν̃ = 10⁷ / λ(nm)

        Args:
            nm: 波长，单位 nm，必须为正数。

        Returns:
            波数，单位 cm⁻¹。

        Raises:
            ComputationError: 波长非正数或非有限时抛出。
        """
        nm = _require_positive(nm, "波长")
        return 1.0e7 / nm

    def cm_inv_to_nm(self, cm_inv: float) -> float:
        """波数 (cm⁻¹) → 波长 (nm).

        公式::

            λ(nm) = 10⁷ / ν̃

        Args:
            cm_inv: 波数，单位 cm⁻¹，必须为正数。

        Returns:
            波长，单位 nm。

        Raises:
            ComputationError: 波数非正数或非有限时抛出。
        """
        cm_inv = _require_positive(cm_inv, "波数")
        return 1.0e7 / cm_inv

    def nm_to_ev(self, nm: float) -> float:
        """波长 (nm) → 光子能量 (eV).

        公式::

            E(eV) = (h·c) / (λ·e) , λ 单位 m

        Args:
            nm: 波长，单位 nm，必须为正数。

        Returns:
            光子能量，单位 eV。

        Raises:
            ComputationError: 波长非正数或非有限时抛出。
        """
        nm = _require_positive(nm, "波长")
        c = self._c
        energy_joule = (c.h * c.c) / (nm * 1.0e-9)
        return energy_joule / c.e

    def ev_to_nm(self, ev: float) -> float:
        """光子能量 (eV) → 波长 (nm).

        公式::

            λ(nm) = (h·c) / (E·e) × 10⁹

        Args:
            ev: 光子能量，单位 eV，必须为正数。

        Returns:
            波长，单位 nm。

        Raises:
            ComputationError: 能量非正数或非有限时抛出。
        """
        ev = _require_positive(ev, "能量")
        c = self._c
        wavelength_m = (c.h * c.c) / (ev * c.e)
        return wavelength_m * 1.0e9

    def nm_to_kj_per_mol(self, nm: float) -> float:
        """波长 (nm) → 摩尔能量 (kJ/mol).

        公式::

            E(kJ/mol) = (h·c·N_A) / (λ·1000) , λ 单位 m

        Args:
            nm: 波长，单位 nm，必须为正数。

        Returns:
            每摩尔光子能量，单位 kJ/mol。

        Raises:
            ComputationError: 波长非正数或非有限时抛出。
        """
        nm = _require_positive(nm, "波长")
        c = self._c
        energy_joule_per_photon = (c.h * c.c) / (nm * 1.0e-9)
        energy_joule_per_mol = energy_joule_per_photon * c.N_A
        return energy_joule_per_mol / 1000.0

    # ------------------------------------------------------------------
    # 通用换算
    # ------------------------------------------------------------------

    def _normalize_unit(self, unit: str) -> str:
        """将单位字符串归一化为规范单位标识."""
        if not isinstance(unit, str):
            raise ComputationError(f"单位必须为字符串, 收到 {type(unit).__name__}")
        key = re.sub(r"\s+", "", unit).lower()
        # 兼容上标 ⁻¹ 与 ^-1 等写法
        key = key.replace("⁻¹", "-1").replace("·", "")
        if key in self._UNIT_ALIASES:
            return self._UNIT_ALIASES[key]
        raise ComputationError(
            f"不支持的单位 {unit!r}, 支持的单位: {self.SUPPORTED_UNITS}"
        )

    def validate_conversion(
        self, value: float, from_unit: str, to_unit: str
    ) -> float:
        """通用单位换算.

        根据源单位与目标单位自动选择换算路径。支持单位别名宽松
        解析（如 ``cm⁻¹``、``eV``、``kJ_per_mol`` 等均可识别）。

        Args:
            value:     待换算的数值。
            from_unit: 源单位标识（nm / cm-1 / ev / kj/mol 或别名）。
            to_unit:   目标单位标识。

        Returns:
            换算后的数值。

        Raises:
            ComputationError: 单位不支持、相同单位无意义换算或数值非法时抛出。

        使用示例::

            >>> uc = UnitConverter()
            >>> round(uc.validate_conversion(500, "nm", "cm-1"), 1)
            20000.0
            >>> round(uc.validate_conversion(2.0, "eV", "nm"), 1)
            620.4
        """
        src = self._normalize_unit(from_unit)
        dst = self._normalize_unit(to_unit)
        if src == dst:
            return _require_finite(value, "待换算数值")

        # 以 nm 为枢纽构建换算表
        to_nm: dict[str, Callable[[float], float]] = {
            "nm": lambda v: _require_finite(v, "待换算数值"),
            "cm-1": self.cm_inv_to_nm,
            "ev": self.ev_to_nm,
            "kj/mol": self._kj_per_mol_to_nm,
        }
        from_nm: dict[str, Callable[[float], float]] = {
            "nm": lambda v: v,
            "cm-1": self.nm_to_cm_inv,
            "ev": self.nm_to_ev,
            "kj/mol": self.nm_to_kj_per_mol,
        }
        if src not in to_nm or dst not in from_nm:
            raise ComputationError(
                f"不支持的换算路径: {from_unit} → {to_unit}"
            )
        nm_value = to_nm[src](value)
        return from_nm[dst](nm_value)

    def _kj_per_mol_to_nm(self, kj_per_mol: float) -> float:
        """摩尔能量 (kJ/mol) → 波长 (nm) 的内部换算."""
        kj_per_mol = _require_positive(kj_per_mol, "摩尔能量")
        c = self._c
        energy_joule_per_photon = (kj_per_mol * 1000.0) / c.N_A
        wavelength_m = (c.h * c.c) / energy_joule_per_photon
        return wavelength_m * 1.0e9


# ============================================================
# JuddOfeltCalculator — Judd-Ofelt 理论计算
# ============================================================


@dataclass(frozen=True)
class JuddOfeltRanges:
    """Judd-Ofelt 强度参数 Ωλ 的合理取值范围（×10⁻²⁰ cm²）.

    与 NumericalLayer 规则 N-R10/N-R11/N-R12 保持一致。

    Attributes:
        omega2_min / omega2_max: Ω₂ 范围（共价性/短程对称性指标）
        omega4_min / omega4_max: Ω₄ 范围（长程/黏度指标）
        omega6_min / omega6_max: Ω₆ 范围（长程/黏度指标）
    """

    omega2_min: float = 1.0
    omega2_max: float = 10.0
    omega4_min: float = 0.5
    omega4_max: float = 5.0
    omega6_min: float = 0.5
    omega6_max: float = 5.0


class JuddOfeltCalculator:
    """Judd-Ofelt 理论计算器.

    实现 Dy3+（及其它稀土离子）发光分析中常用的 Judd-Ofelt 理论
    相关计算：振子强度、辐射跃迁速率、分支比、参数校验与量子效率
    评估。

    使用示例::

        >>> jo = JuddOfeltCalculator()
        >>> jo.validate_judd_ofelt_params(3.0, 1.5, 1.0)
        True
        >>> jo.calculate_branching_ratio(120.0, 400.0)
        0.3
    """

    def __init__(
        self,
        constants: PhysicalConstants | None = None,
        ranges: JuddOfeltRanges | None = None,
    ) -> None:
        self._c = constants or CONST
        self._ranges = ranges or JuddOfeltRanges()

    # ------------------------------------------------------------------
    # 振子强度与辐射速率
    # ------------------------------------------------------------------

    def calculate_oscillator_strength(
        self, tau_rad: float, lambda_p: float, n: float, J: int
    ) -> float:
        """根据辐射寿命计算电偶极振子强度.

        采用辐射速率与振子强度的 SI 关系式（由 Einstein A 系数与
        吸收振子强度的关系 ``A = ω²e²/(2πε₀m_ec³)·(g_l/g_u)·f``
        反解，并纳入折射率修正 ``1/n²`` 与初态简并度 ``1/(2J+1)``）::

            f = (ε₀ · m_e · c · λ² · A_rad) / (2π · e² · n² · (2J+1))

        其中 ``A_rad = 1/τ_rad`` 为辐射跃迁速率，``λ`` 为峰值发射
        波长（nm → m 换算），``n`` 为介质折射率，``J`` 为辐射初态
        总角动量量子数， ``(2J+1)`` 为初态简并度。

        Args:
            tau_rad:  辐射寿命，单位 s，必须为正数。
            lambda_p: 峰值发射波长，单位 nm，必须为正数。
            n:        介质折射率，必须为正数。
            J:        辐射初态总角动量量子数，非负整数。

        Returns:
            振子强度 f（无量纲，典型量级 10⁻⁷–10⁻²）。

        Raises:
            ComputationError: 参数非法时抛出。
        """
        tau_rad = _require_positive(tau_rad, "辐射寿命")
        lambda_p = _require_positive(lambda_p, "峰值波长")
        n = _require_positive(n, "折射率")
        if not isinstance(J, int) or isinstance(J, bool) or J < 0:
            raise ComputationError(f"角动量量子数 J 必须为非负整数, 收到 {J!r}")

        c = self._c
        a_rad = 1.0 / tau_rad
        lambda_m = lambda_p * 1.0e-9
        degeneracy = 2 * J + 1
        numerator = c.epsilon_0 * c.m_e * c.c * (lambda_m ** 2) * a_rad
        denominator = 2.0 * math.pi * (c.e ** 2) * (n ** 2) * degeneracy
        return numerator / denominator

    def calculate_radiative_rate(self, tau_rad_inv: float) -> float:
        """计算辐射跃迁速率.

        辐射跃迁速率 ``A_rad`` 等于辐射寿命的倒数。本方法将入参
        ``tau_rad_inv``（即 ``1/τ_rad``，单位 s⁻¹）作为辐射寿命的
        倒数，校验后返回辐射跃迁速率。

        Args:
            tau_rad_inv: 辐射寿命的倒数 (1/τ_rad)，单位 s⁻¹，必须为正数。

        Returns:
            辐射跃迁速率 A_rad，单位 s⁻¹。

        Raises:
            ComputationError: 数值非正或非有限时抛出。
        """
        return _require_positive(tau_rad_inv, "辐射寿命倒数 (1/τ_rad)")

    def calculate_branching_ratio(self, a_jj: float, total: float) -> float:
        """计算分支比.

        公式::

            β = A_jj / A_total

        Args:
            a_jj:  指定跃迁的辐射跃迁速率，单位 s⁻¹。
            total: 所有相关跃迁的辐射速率之和，单位 s⁻¹，必须为正数。

        Returns:
            分支比 β，取值 [0, 1]。

        Raises:
            ComputationError: 总速率非正、分支速率超出总速率或参数非法时抛出。
        """
        a_jj = _require_finite(a_jj, "分支跃迁速率")
        total = _require_positive(total, "总跃迁速率")
        if a_jj < 0.0:
            raise ComputationError(f"分支跃迁速率不能为负, 收到 {a_jj}")
        if a_jj > total:
            raise ComputationError(
                f"分支速率 {a_jj} 大于总速率 {total}, 数据不一致"
            )
        return a_jj / total

    # ------------------------------------------------------------------
    # 参数校验与量子效率
    # ------------------------------------------------------------------

    def validate_judd_ofelt_params(
        self, omega2: float, omega4: float, omega6: float
    ) -> bool:
        """校验 Judd-Ofelt 强度参数 Ωλ 是否在合理范围内.

        合理范围（×10⁻²⁰ cm²）::

            Ω₂ ∈ [1.0, 10.0]
            Ω₄ ∈ [0.5,  5.0]
            Ω₆ ∈ [0.5,  5.0]

        与 NumericalLayer 规则 N-R10/N-R11/N-R12 保持一致。

        Args:
            omega2: Ω₂ 参数值。
            omega4: Ω₄ 参数值。
            omega6: Ω₆ 参数值。

        Returns:
            全部在范围内返回 ``True``，否则返回 ``False``。
        """
        r = self._ranges
        try:
            o2 = _require_finite(omega2, "Ω₂")
            o4 = _require_finite(omega4, "Ω₄")
            o6 = _require_finite(omega6, "Ω₆")
        except ComputationError:
            return False
        return (
            r.omega2_min <= o2 <= r.omega2_max
            and r.omega4_min <= o4 <= r.omega4_max
            and r.omega6_min <= o6 <= r.omega6_max
        )

    def calculate_qe_from_jo(
        self,
        omega2: float,
        omega4: float,
        omega6: float,
        tau_rad: float,
        tau_obs: float,
    ) -> float:
        """基于 Judd-Ofelt 参数与寿命计算量子效率 (QE).

        量子效率定义为辐射速率占总跃迁速率的比例::

            QE = A_rad / (A_rad + A_nr) = τ_obs / τ_rad

        其中 ``A_rad = 1/τ_rad``，``A_total = 1/τ_obs``，
        ``A_nr = A_total - A_rad``。计算前先对 Ωλ 参数进行合理性
        校验，若参数超出物理合理范围则抛出异常，以避免基于不合理
        输入得出误导性结果。

        Args:
            omega2:  Ω₂ 参数值（×10⁻²⁰ cm²）。
            omega4:  Ω₄ 参数值（×10⁻²⁰ cm²）。
            omega6:  Ω₆ 参数值（×10⁻²⁰ cm²）。
            tau_rad: 辐射寿命，单位 s，必须为正数。
            tau_obs: 观测寿命，单位 s，必须为正数。

        Returns:
            量子效率 QE，取值 [0, 1]。

        Raises:
            ComputationError: Ωλ 参数超出范围、寿命非法或 τ_obs > τ_rad
                时抛出。
        """
        if not self.validate_judd_ofelt_params(omega2, omega4, omega6):
            raise ComputationError(
                f"Judd-Ofelt 参数超出合理范围: "
                f"Ω₂={omega2}, Ω₄={omega4}, Ω₆={omega6}"
            )
        tau_rad = _require_positive(tau_rad, "辐射寿命")
        tau_obs = _require_positive(tau_obs, "观测寿命")
        if tau_obs > tau_rad:
            raise ComputationError(
                f"观测寿命 ({tau_obs}s) 不应大于辐射寿命 ({tau_rad}s)"
            )
        qe = tau_obs / tau_rad
        # 数值钳制，避免浮点误差导致略微越界
        return max(0.0, min(1.0, qe))


# ============================================================
# CIECalculator — CIE 1931 色度学计算
# ============================================================


@dataclass(frozen=True)
class CIESpectrumPoint:
    """CIE 1931 光谱轨迹采样点.

    Attributes:
        wavelength: 波长，单位 nm。
        x: CIE 1931 色品坐标 x 分量。
        y: CIE 1931 色品坐标 y 分量。
    """

    wavelength: float
    x: float
    y: float


#: CIE 1931 2° 标准观察者光谱轨迹（10 nm 采样，380–700 nm）
_CIE_SPECTRUM_LOCUS: tuple[CIESpectrumPoint, ...] = (
    CIESpectrumPoint(380, 0.1741, 0.0050),
    CIESpectrumPoint(390, 0.1740, 0.0050),
    CIESpectrumPoint(400, 0.1733, 0.0048),
    CIESpectrumPoint(410, 0.1726, 0.0048),
    CIESpectrumPoint(420, 0.1714, 0.0051),
    CIESpectrumPoint(430, 0.1689, 0.0069),
    CIESpectrumPoint(440, 0.1644, 0.0109),
    CIESpectrumPoint(450, 0.1566, 0.0177),
    CIESpectrumPoint(460, 0.1440, 0.0297),
    CIESpectrumPoint(470, 0.1241, 0.0578),
    CIESpectrumPoint(480, 0.0913, 0.1327),
    CIESpectrumPoint(490, 0.0454, 0.2950),
    CIESpectrumPoint(500, 0.0082, 0.5384),
    CIESpectrumPoint(510, 0.0139, 0.7502),
    CIESpectrumPoint(520, 0.0743, 0.8338),
    CIESpectrumPoint(530, 0.1547, 0.8059),
    CIESpectrumPoint(540, 0.2296, 0.7543),
    CIESpectrumPoint(550, 0.3016, 0.6923),
    CIESpectrumPoint(560, 0.3731, 0.6245),
    CIESpectrumPoint(570, 0.4441, 0.5547),
    CIESpectrumPoint(580, 0.5125, 0.4866),
    CIESpectrumPoint(590, 0.5752, 0.4242),
    CIESpectrumPoint(600, 0.6270, 0.3725),
    CIESpectrumPoint(610, 0.6658, 0.3340),
    CIESpectrumPoint(620, 0.6915, 0.3083),
    CIESpectrumPoint(630, 0.7079, 0.2920),
    CIESpectrumPoint(640, 0.7190, 0.2809),
    CIESpectrumPoint(650, 0.7260, 0.2740),
    CIESpectrumPoint(660, 0.7300, 0.2700),
    CIESpectrumPoint(670, 0.7320, 0.2680),
    CIESpectrumPoint(680, 0.7334, 0.2666),
    CIESpectrumPoint(690, 0.7344, 0.2656),
    CIESpectrumPoint(700, 0.7347, 0.2653),
)


@dataclass(frozen=True)
class CIEWhitePoint:
    """CIE 标准白点.

    Attributes:
        name: 白点名称。
        x: 色品坐标 x。
        y: 色品坐标 y。
        Xn: XYZ 三刺激值 X（归一化 Y=100）。
        Yn: XYZ 三刺激值 Y。
        Zn: XYZ 三刺激值 Z。
    """

    name: str
    x: float
    y: float
    Xn: float
    Yn: float
    Zn: float


#: CIE 标准照明体 D65
_D65: CIEWhitePoint = CIEWhitePoint(
    name="D65", x=0.3127, y=0.3290, Xn=95.047, Yn=100.0, Zn=108.883
)


class CIECalculator:
    """CIE 1931 色度学计算器.

    提供主波长、色纯度、相关色温 (CCT)、色坐标合法性校验及
    CIE1976 色差 (ΔE_ab) 的近似计算。主波长与色纯度基于从标准
    白点出发、与光谱轨迹的射线求交实现。

    使用示例::

        >>> cie = CIECalculator()
        >>> 0 < cie.calculate_cct(0.42, 0.40) < 10000
        True
        >>> cie.validate_cie_coordinates(0.42, 0.46)
        True
    """

    def __init__(self, white_point: CIEWhitePoint | None = None) -> None:
        self._wp = white_point or _D65
        self._locus: tuple[CIESpectrumPoint, ...] = _CIE_SPECTRUM_LOCUS

    # ------------------------------------------------------------------
    # 内部几何工具
    # ------------------------------------------------------------------

    @staticmethod
    def _ray_segment_intersection(
        px: float, py: float, dx: float, dy: float,
        ax: float, ay: float, bx: float, by: float,
    ) -> tuple[float, float] | None:
        """求射线 (P + t·D, t∈ℝ) 与线段 AB 的交点参数 (t, u).

        Args:
            px, py: 射线起点。
            dx, dy: 射线方向向量。
            ax, ay: 线段端点 A。
            bx, by: 线段端点 B。

        Returns:
            (t, u) 当且仅当存在交点且 u∈[0,1]；否则返回 ``None``。
        """
        ex = bx - ax
        ey = by - ay
        denom = dx * (-ey) - (-ex) * dy
        if abs(denom) < 1.0e-12:
            return None  # 平行
        rx = ax - px
        ry = ay - py
        t = (rx * (-ey) - (-ex) * ry) / denom
        u = (dx * ry - dy * rx) / denom
        if -1.0e-9 <= u <= 1.0 + 1.0e-9:
            return t, u
        return None

    def _dominant_intersection(
        self, x: float, y: float
    ) -> tuple[float, float, float] | None:
        """计算从白点出发经过 (x, y) 的射线与光谱轨迹的交点.

        Returns:
            (wavelength, ix, iy) 交点波长与坐标；若无交点返回 ``None``。
            ``wavelength`` 为负表示互补波长（紫线区域）。
        """
        wx, wy = self._wp.x, self._wp.y
        dx, dy = x - wx, y - wy
        if abs(dx) < 1.0e-12 and abs(dy) < 1.0e-12:
            return None

        # 1) 前向 (t>0) 与光谱轨迹求交
        best: tuple[float, float, float, float] | None = None  # (t, wl, ix, iy)
        for i in range(len(self._locus) - 1):
            a = self._locus[i]
            b = self._locus[i + 1]
            res = self._ray_segment_intersection(
                wx, wy, dx, dy, a.x, a.y, b.x, b.y
            )
            if res is None:
                continue
            t, u = res
            if t > 1.0e-9:
                wl = a.wavelength + u * (b.wavelength - a.wavelength)
                ix = a.x + u * (b.x - a.x)
                iy = a.y + u * (b.y - a.y)
                if best is None or t < best[0]:
                    best = (t, wl, ix, iy)
        if best is not None:
            return best[1], best[2], best[3]

        # 2) 紫线区域：前向与紫线（首末点连线）求交，反向求互补波长
        first = self._locus[0]
        last = self._locus[-1]
        purple = self._ray_segment_intersection(
            wx, wy, dx, dy, last.x, last.y, first.x, first.y
        )
        if purple is None or purple[0] <= 1.0e-9:
            return None
        # 反向射线 (-D) 与光谱轨迹求交，取最近交点为互补波长
        rev_best: tuple[float, float] | None = None  # (|t|, wl)
        for i in range(len(self._locus) - 1):
            a = self._locus[i]
            b = self._locus[i + 1]
            res = self._ray_segment_intersection(
                wx, wy, -dx, -dy, a.x, a.y, b.x, b.y
            )
            if res is None:
                continue
            t, u = res
            if t > 1.0e-9:
                wl = a.wavelength + u * (b.wavelength - a.wavelength)
                if rev_best is None or t < rev_best[0]:
                    rev_best = (t, wl)
        if rev_best is not None:
            # 返回负波长表示互补；坐标用紫线交点
            _, u_p = purple
            ix = last.x + u_p * (first.x - last.x)
            iy = last.y + u_p * (first.y - last.y)
            return -rev_best[1], ix, iy
        return None

    # ------------------------------------------------------------------
    # 色度计算
    # ------------------------------------------------------------------

    def calculate_dominant_wavelength(self, x: float, y: float) -> float:
        """由 CIE 色品坐标近似计算主波长.

        从标准白点 (D65) 出发，经过样本点 (x, y) 作射线，与 CIE 1931
        光谱轨迹的交点对应波长即为主波长。若样本点位于紫线区域，
        则返回互补波长（以负值表示，按色度学惯例）。

        Args:
            x: 色品坐标 x。
            y: 色品坐标 y。

        Returns:
            主波长，单位 nm；紫线区域返回负的互补波长。

        Raises:
            ComputationError: 坐标非法或无法确定主波长时抛出。
        """
        x = _require_finite(x, "CIE x")
        y = _require_finite(y, "CIE y")
        res = self._dominant_intersection(x, y)
        if res is None:
            raise ComputationError(
                f"无法由坐标 ({x}, {y}) 确定主波长，可能坐标越界或重合于白点"
            )
        return res[0]

    def calculate_purity(self, x: float, y: float) -> float:
        """计算色纯度（激发纯度, excitation purity）.

        公式::

            Pe = |W - S| / |W - L|

        其中 W 为白点，S 为样本点，L 为主波长在光谱轨迹上的对应点
        （紫线区域则为紫线交点）。

        Args:
            x: 色品坐标 x。
            y: 色品坐标 y。

        Returns:
            色纯度 Pe，取值 [0, 1]。

        Raises:
            ComputationError: 坐标非法或无法计算时抛出。
        """
        x = _require_finite(x, "CIE x")
        y = _require_finite(y, "CIE y")
        res = self._dominant_intersection(x, y)
        if res is None:
            raise ComputationError(
                f"无法由坐标 ({x}, {y}) 计算色纯度"
            )
        _, lx, ly = res
        wx, wy = self._wp.x, self._wp.y
        num = math.hypot(x - wx, y - wy)
        den = math.hypot(lx - wx, ly - wy)
        if den < 1.0e-12:
            return 0.0
        purity = num / den
        return max(0.0, min(1.0, purity))

    def calculate_cct(self, x: float, y: float) -> float:
        """使用 McCamy 公式近似计算相关色温 (CCT).

        公式::

            n  = (x - xe) / (ye - y)
            CCT = 449 n³ + 3525 n² + 6823.3 n + 5520.33

        其中 ``(xe, ye) = (0.3320, 0.1858)`` 为参考点。

        Args:
            x: 色品坐标 x。
            y: 色品坐标 y。

        Returns:
            相关色温，单位 K。

        Raises:
            ComputationError: 坐标非法或导致除零时抛出。
        """
        x = _require_finite(x, "CIE x")
        y = _require_finite(y, "CIE y")
        xe, ye = 0.3320, 0.1858
        denom = ye - y
        if abs(denom) < 1.0e-12:
            raise ComputationError(f"CCT 计算分母为零 (y≈{y})")
        n = (x - xe) / denom
        cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
        if not math.isfinite(cct):
            raise ComputationError(f"CCT 计算结果非有限: {cct}")
        return cct

    def validate_cie_coordinates(self, x: float, y: float) -> bool:
        """校验色品坐标是否位于 CIE 1931 色域内.

        通过对光谱轨迹闭合多边形（含紫线）进行射线法点内判定，
        并附加 ``x>=0, y>=0, x+y<=1`` 的基本约束。

        Args:
            x: 色品坐标 x。
            y: 色品坐标 y。

        Returns:
            位于色域内返回 ``True``，否则返回 ``False``。
        """
        try:
            x = _require_finite(x, "CIE x")
            y = _require_finite(y, "CIE y")
        except ComputationError:
            return False
        if x < 0.0 or y < 0.0 or (x + y) > 1.0 + 1.0e-9:
            return False
        # 构建闭合多边形：光谱轨迹 + 紫线（末点回到首点）
        poly = [(p.x, p.y) for p in self._locus]
        return self._point_in_polygon(x, y, poly)

    @staticmethod
    def _point_in_polygon(
        px: float, py: float, polygon: list[tuple[float, float]]
    ) -> bool:
        """射线法判断点是否在多边形内部."""
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1.0e-15) + xi
            ):
                inside = not inside
            j = i
        return inside

    def calculate_color_difference(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """计算 CIE1976 色差 (ΔE_ab).

        将两组色品坐标 (假定等亮度 Y=1) 转换为 XYZ，再转换至
        CIE1976 L*a*b* 颜色空间，计算欧氏距离::

            ΔE = sqrt(ΔL² + Δa² + Δb²)

        Args:
            x1, y1: 第一组色品坐标。
            x2, y2: 第二组色品坐标。

        Returns:
            CIE1976 色差 ΔE。

        Raises:
            ComputationError: 坐标非法或导致除零时抛出。
        """
        l1, a1, b1 = self._xy_to_lab(x1, y1)
        l2, a2, b2 = self._xy_to_lab(x2, y2)
        return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

    def _xy_to_lab(self, x: float, y: float) -> tuple[float, float, float]:
        """色品坐标 (x, y) → CIE1976 L*a*b*（假定 Y=1, D65 参考白）."""
        x = _require_finite(x, "CIE x")
        y = _require_finite(y, "CIE y")
        if abs(y) < 1.0e-12:
            raise ComputationError(f"色品坐标 y 过小 (y={y}), 无法转换")
        # xy → XYZ (Y=100)
        xr = x / y
        zr = (1.0 - x - y) / y
        X = xr * 100.0
        Y = 100.0
        Z = zr * 100.0
        wp = self._wp
        fx = self._lab_f(X / wp.Xn)
        fy = self._lab_f(Y / wp.Yn)
        fz = self._lab_f(Z / wp.Zn)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return L, a, b

    @staticmethod
    def _lab_f(t: float) -> float:
        """CIE1976 Lab 转换的非线性函数 f(t)."""
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3.0 * delta * delta) + 4.0 / 29.0


# ============================================================
# ErrorAnalyzer — 误差传播与不确定度分析
# ============================================================


class ErrorAnalyzer:
    """误差传播与不确定度分析器.

    实现不确定度传播、离群点检测（IQR / Z-score）、相对误差与
    置信区间计算，遵循 GUM《测量不确定度表示指南》的基本方法。

    使用示例::

        >>> ea = ErrorAnalyzer()
        >>> ea.calculate_relative_error(10.5, 10.0)
        0.05
        >>> ea.detect_outliers_iqr([1, 2, 2, 3, 3, 100])
        [5]
    """

    #: 支持的不确定度传播运算
    _OPERATIONS: dict[str, str] = {
        "add": "add",
        "+": "add",
        "加": "add",
        "subtract": "sub",
        "sub": "sub",
        "-": "sub",
        "减": "sub",
        "multiply": "mul",
        "mul": "mul",
        "*": "mul",
        "乘": "mul",
        "divide": "div",
        "div": "div",
        "/": "div",
        "除": "div",
        "power": "pow",
        "pow": "pow",
        "**": "pow",
        "^": "pow",
        "幂": "pow",
    }

    # ------------------------------------------------------------------
    # 不确定度传播
    # ------------------------------------------------------------------

    def propagate_uncertainty(
        self,
        value: float,
        uncertainty: float,
        operation: str,
        operand: float,
    ) -> tuple[float, float]:
        """通过指定运算传播不确定度.

        假定 ``operand`` 为精确常数（不确定度为零），依据一阶
        泰勒展开（线性传播）计算结果值与合成不确定度：

        - 加减: ``y = v ± o``,  ``u_y = u_v``
        - 乘:   ``y = v · o``,  ``u_y = |o| · u_v``
        - 除:   ``y = v / o``,  ``u_y = u_v / |o|``
        - 幂:   ``y = v^o``,    ``u_y = |o · v^(o-1)| · u_v``

        Args:
            value:       输入量值。
            uncertainty: 输入量的不确定度（非负）。
            operation:   运算类型（add/sub/mul/div/pow 或别名）。
            operand:     运算操作数（视为精确值）。

        Returns:
            ``(结果值, 合成不确定度)``。

        Raises:
            ComputationError: 运算不支持、不确定度为负或运算非法
                （如除零、负底数取实幂）时抛出。
        """
        v = _require_finite(value, "输入量值")
        u = _require_finite(uncertainty, "不确定度")
        if u < 0.0:
            raise ComputationError(f"不确定度不能为负, 收到 {u}")
        o = _require_finite(operand, "操作数")

        key = self._normalize_operation(operation)
        if key == "add":
            return v + o, u
        if key == "sub":
            return v - o, u
        if key == "mul":
            return v * o, abs(o) * u
        if key == "div":
            if abs(o) < 1.0e-15:
                raise ComputationError("除法运算的操作数不能为零")
            return v / o, u / abs(o)
        if key == "pow":
            if v < 0.0 and not float(o).is_integer():
                raise ComputationError(
                    f"负底数 ({v}) 的非整数幂 ({o}) 在实数域无定义"
                )
            if v == 0.0 and o <= 0.0:
                raise ComputationError(
                    f"零底数的非正幂 ({o}) 无定义"
                )
            result = v ** o
            derivative = abs(o * (v ** (o - 1))) if v != 0.0 else 0.0
            return result, derivative * u
        raise ComputationError(f"不支持的运算: {operation!r}")

    def _normalize_operation(self, operation: str) -> str:
        """归一化运算标识."""
        if not isinstance(operation, str):
            raise ComputationError(
                f"运算类型必须为字符串, 收到 {type(operation).__name__}"
            )
        key = re.sub(r"\s+", "", operation).lower()
        if key not in self._OPERATIONS:
            raise ComputationError(
                f"不支持的运算 {operation!r}, 支持: add/sub/mul/div/pow"
            )
        return self._OPERATIONS[key]

    # ------------------------------------------------------------------
    # 离群点检测
    # ------------------------------------------------------------------

    def detect_outliers_iqr(self, values: list[float]) -> list[int]:
        """使用 IQR（四分位距）法检测离群点.

        以第一四分位 Q1 与第三四分位 Q3 定义内围栏::

            下界 = Q1 - 1.5 · IQR
            上界 = Q3 + 1.5 · IQR

        超出内围栏的样本视为离群点。

        Args:
            values: 数值列表（至少 4 个有效值以保证分位数稳健）。

        Returns:
            离群点在原列表中的索引列表（按升序）。
        """
        data = self._clean_values(values)
        if len(data) < 4:
            return []
        # 仅取有限数值并保留原索引，避免 NaN/Inf 破坏排序
        indexed = [
            (i, float(v))
            for i, v in enumerate(values)
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        ]
        indexed.sort(key=lambda kv: kv[1])
        order_index = [idx for idx, _ in indexed]
        sorted_vals = [v for _, v in indexed]
        q1 = _quantile(sorted_vals, 0.25)
        q3 = _quantile(sorted_vals, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [
            order_index[i]
            for i, v in enumerate(sorted_vals)
            if v < lower or v > upper
        ]
        return sorted(outliers)

    def detect_outliers_zscore(
        self, values: list[float], threshold: float = 3.0
    ) -> list[int]:
        """使用 Z-score 法检测离群点.

        公式::

            z_i = (x_i - μ) / σ

        ``|z_i| > threshold`` 的样本视为离群点（默认阈值 3.0）。
        标准差采用样本标准差 (n-1)。

        Args:
            values:    数值列表。
            threshold: Z-score 阈值，必须为正数。

        Returns:
            离群点索引列表（按升序）。

        Raises:
            ComputationError: 阈值非正或样本不足时抛出。
        """
        if threshold <= 0.0:
            raise ComputationError(f"Z-score 阈值必须为正, 收到 {threshold}")
        data = self._clean_values(values)
        if len(data) < 3:
            return []
        mean = statistics.fmean(data)
        try:
            std = statistics.stdev(data)
        except statistics.StatisticsError:
            return []
        if std < 1.0e-15:
            return []
        outliers = [
            i for i, v in enumerate(values)
            if isinstance(v, (int, float)) and math.isfinite(float(v))
            and abs((float(v) - mean) / std) > threshold
        ]
        return sorted(outliers)

    # ------------------------------------------------------------------
    # 相对误差与置信区间
    # ------------------------------------------------------------------

    def calculate_relative_error(
        self, measured: float, true_value: float
    ) -> float:
        """计算相对误差.

        公式::

            δ = |measured - true_value| / |true_value|

        Args:
            measured:   测量值。
            true_value: 真值。

        Returns:
            相对误差（非负，无量纲）。

        Raises:
            ComputationError: 真值为零或参数非法时抛出。
        """
        m = _require_finite(measured, "测量值")
        t = _require_finite(true_value, "真值")
        if abs(t) < 1.0e-15:
            raise ComputationError("真值为零，相对误差无定义")
        return abs(m - t) / abs(t)

    def calculate_confidence_interval(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """计算样本均值的置信区间.

        采用样本均值与样本标准差，并通过 Cornish-Fisher 展开近似
        t 分布临界值，置信区间为::

            CI = mean ± t* · (s / √n)

        Args:
            values:    数值列表（至少 2 个有效值）。
            confidence: 置信水平，取值 (0, 1)，默认 0.95。

        Returns:
            ``(下限, 上限)``。

        Raises:
            ComputationError: 置信水平非法、样本不足或标准差为零时抛出。
        """
        if not 0.0 < confidence < 1.0:
            raise ComputationError(
                f"置信水平必须在 (0, 1) 内, 收到 {confidence}"
            )
        data = self._clean_values(values)
        n = len(data)
        if n < 2:
            raise ComputationError(
                f"置信区间计算至少需要 2 个有效值, 收到 {n} 个"
            )
        mean = statistics.fmean(data)
        try:
            std = statistics.stdev(data)
        except statistics.StatisticsError as exc:
            raise ComputationError("无法计算样本标准差") from exc
        if std < 1.0e-15:
            # 所有值相同，区间退化为点
            return mean, mean
        df = n - 1
        t_crit = self._t_critical(confidence, df)
        half_width = t_crit * std / math.sqrt(n)
        return mean - half_width, mean + half_width

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_values(values: list[float]) -> list[float]:
        """清洗数值列表：过滤非有限数值并校验类型."""
        if not isinstance(values, list):
            raise ComputationError(
                f"输入必须为列表, 收到 {type(values).__name__}"
            )
        cleaned: list[float] = []
        for v in values:
            if isinstance(v, bool):
                continue
            if not isinstance(v, (int, float)):
                raise ComputationError(
                    f"列表元素必须为数值, 收到 {v!r} ({type(v).__name__})"
                )
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        return cleaned

    @staticmethod
    def _t_critical(confidence: float, df: int) -> float:
        """通过 Cornish-Fisher 展开近似 t 分布临界值 t*.

        Args:
            confidence: 置信水平。
            df:         自由度 (n-1)。

        Returns:
            双侧 t 临界值。
        """
        alpha = 1.0 - confidence
        p = 1.0 - alpha / 2.0
        z = NormalDist().inv_cdf(p)
        # Cornish-Fisher 展开（以自由度 df 为参数）
        z3 = z ** 3
        z5 = z ** 5
        z7 = z ** 7
        t = (
            z
            + (z3 + z) / (4.0 * df)
            + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df)
            + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
            / (384.0 * df ** 3)
        )
        return t


# ============================================================
# 模块公开接口
# ============================================================


__all__ = [
    "PhysicalConstants",
    "ComputationError",
    "UnitConverter",
    "JuddOfeltRanges",
    "JuddOfeltCalculator",
    "CIESpectrumPoint",
    "CIEWhitePoint",
    "CIECalculator",
    "ErrorAnalyzer",
    "CONST",
]
