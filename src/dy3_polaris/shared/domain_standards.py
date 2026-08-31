"""领域标准值库 — 仅接收可核验、适用范围明确的数值标准。

知识点标准值 (数值/容差/单位/来源) 统一在此定义, 供:
- L3 FactChecker 事实校验 (l5/unified_app 运行时构建)
- L3 测试 fixture 与各层复用
- 后续 L2 练习/考核判题引用

当前语料没有满足上述条件的生产级记录，因此生产库保持为空。论文中
观察到的波长、效率或工艺参数属于特定材料/条件下的证据，不是可用于
校验所有回答的全局标准值。
"""
from __future__ import annotations

from typing import Any

from dy3_polaris.l3.fact_check import (
    StandardValue,
    StandardValueStore,
    ToleranceType,
)

#: kp_id -> param_name -> (value, tolerance, tolerance_type, unit, source_ref)
#: New entries require a real citation and an explicit material/method scope.
DOMAIN_STANDARD_VALUES: dict[
    str,
    dict[str, tuple[float, float, ToleranceType, str, str]],
] = {}


def build_domain_standard_store() -> StandardValueStore:
    """构建加载全部已核验领域标准值的标准值库 (单点入口).

    Returns:
        已填充 DOMAIN_STANDARD_VALUES 的 StandardValueStore.
    """
    store = StandardValueStore()
    for kp_id, params in DOMAIN_STANDARD_VALUES.items():
        for param, (value, tol, tol_type, unit, ref) in params.items():
            store.add(
                StandardValue(
                    kp_id=kp_id,
                    param_name=param,
                    standard_value=value,
                    tolerance=tol,
                    tolerance_type=tol_type,
                    unit=unit,
                    source_ref=ref,
                )
            )
    return store


def domain_standard_count() -> int:
    """领域标准值总数."""
    return sum(len(params) for params in DOMAIN_STANDARD_VALUES.values())


def list_domain_standards() -> list[dict[str, Any]]:
    """领域标准值清单 (便于展示/调试)."""
    return [
        {
            "kp_id": kp_id,
            "param_name": param,
            "standard_value": spec[0],
            "tolerance": spec[1],
            "tolerance_type": spec[2].value,
            "unit": spec[3],
            "source_ref": spec[4],
        }
        for kp_id, params in DOMAIN_STANDARD_VALUES.items()
        for param, spec in params.items()
    ]


__all__ = [
    "DOMAIN_STANDARD_VALUES",
    "build_domain_standard_store",
    "domain_standard_count",
    "list_domain_standards",
]
