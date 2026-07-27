"""MCP 工具注册中心.

管理 Dy3+ Polaris 系统的 47 个 MCP 工具：
- 11 个内部计算工具 (diagnosis/review/guidance/shared)
- 20 个 L3 领域知识连接器 (Tier-1/2/3)
- 11 个 L2 Skillbook 教学技能
- 5 个外部工具详细实现

核心组件：
- ToolRegistry: 中心注册表（注册/发现/校验/依赖解析）
- SchemaValidator: JSON Schema 校验器
- 内置工具定义模块: internal_tools / connector_tools / skillbook_tools / external_tools

快速使用:
    from dy3_polaris.l6.registry import get_registry, load_all_tools

    registry = get_registry()
    load_all_tools(registry)  # 加载全部 47 个工具

    # 按分类查找
    internal = registry.discover_by_category(ToolCategory.INTERNAL)

    # 导出 MCP 兼容列表
    mcp_list = registry.export_mcp_tool_list()
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.models import Dy3ToolAnnotations, LayerTag, ToolCategory, ToolRegistration
from .schema_validator import (
    SchemaValidator,
    get_validator,
    reset_validator,
)
from .tool_registry import (
    ToolEntry,
    ToolRegistry,
    get_registry,
    reset_registry,
)
from .internal_tools import (
    INTERNAL_TOOL_DEFINITIONS,
    INTERNAL_TOOL_NAMES,
    DIAGNOSIS_TOOLS,
    REVIEW_TOOLS,
    GUIDANCE_TOOLS,
    SHARED_TOOLS,
    get_internal_tool,
)
from .connector_tools import (
    CONNECTOR_TOOL_DEFINITIONS,
    CONNECTOR_TOOL_NAMES,
    TIER1_TOOLS,
    TIER2_TOOLS,
    TIER3_TOOLS,
    get_connector_tool,
)
from .skillbook_tools import (
    SKILLBOOK_TOOL_DEFINITIONS,
    SKILLBOOK_TOOL_NAMES,
    get_skillbook_tool,
)
from .external_tools import (
    EXTERNAL_TOOL_DEFINITIONS,
    EXTERNAL_TOOL_NAMES,
    get_external_tool,
)


# ============================================================
# 全部工具定义汇总
# ============================================================

ALL_TOOL_DEFINITIONS: list[tuple[ToolRegistration, Any]] = (
    INTERNAL_TOOL_DEFINITIONS
    + CONNECTOR_TOOL_DEFINITIONS
    + SKILLBOOK_TOOL_DEFINITIONS
    + EXTERNAL_TOOL_DEFINITIONS
)

ALL_TOOL_NAMES: list[str] = [reg.name for reg, _ in ALL_TOOL_DEFINITIONS]

# 总数
TOTAL_TOOL_COUNT: int = len(ALL_TOOL_DEFINITIONS)


# ============================================================
# 批量加载函数
# ============================================================

def load_all_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """将全部 47 个工具加载到注册中心.

    Args:
        registry: 目标注册中心，None 使用全局单例

    Returns:
        加载完成的 ToolRegistry
    """
    reg = registry or get_registry()
    reg.register_batch_sync(ALL_TOOL_DEFINITIONS)
    return reg


def load_internal_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """仅加载 11 个内部工具."""
    reg = registry or get_registry()
    reg.register_batch_sync(INTERNAL_TOOL_DEFINITIONS)
    return reg


def load_connector_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """仅加载 20 个连接器工具."""
    reg = registry or get_registry()
    reg.register_batch_sync(CONNECTOR_TOOL_DEFINITIONS)
    return reg


def load_skillbook_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """仅加载 11 个 Skillbook 技能工具."""
    reg = registry or get_registry()
    reg.register_batch_sync(SKILLBOOK_TOOL_DEFINITIONS)
    return reg


def load_external_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """仅加载 5 个外部工具."""
    reg = registry or get_registry()
    reg.register_batch_sync(EXTERNAL_TOOL_DEFINITIONS)
    return reg


# ============================================================
# 工具查找便捷函数
# ============================================================

def find_tool(name: str) -> tuple[ToolRegistration, Any] | None:
    """在全部 47 个工具中查找.

    Returns:
        (registration, handler) 或 None
    """
    for reg, handler in ALL_TOOL_DEFINITIONS:
        if reg.name == name:
            return reg, handler
    return None


def get_tool_names_by_category(category: ToolCategory) -> list[str]:
    """按分类获取工具名列表."""
    return [reg.name for reg, _ in ALL_TOOL_DEFINITIONS if reg.annotations.category == category]


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 核心类
    "ToolRegistry",
    "ToolEntry",
    "SchemaValidator",
    # 单例
    "get_registry",
    "reset_registry",
    "get_validator",
    "reset_validator",
    # 工具定义
    "ALL_TOOL_DEFINITIONS",
    "ALL_TOOL_NAMES",
    "TOTAL_TOOL_COUNT",
    "INTERNAL_TOOL_DEFINITIONS",
    "INTERNAL_TOOL_NAMES",
    "CONNECTOR_TOOL_DEFINITIONS",
    "CONNECTOR_TOOL_NAMES",
    "SKILLBOOK_TOOL_DEFINITIONS",
    "SKILLBOOK_TOOL_NAMES",
    "EXTERNAL_TOOL_DEFINITIONS",
    "EXTERNAL_TOOL_NAMES",
    # 子分类
    "DIAGNOSIS_TOOLS",
    "REVIEW_TOOLS",
    "GUIDANCE_TOOLS",
    "SHARED_TOOLS",
    "TIER1_TOOLS",
    "TIER2_TOOLS",
    "TIER3_TOOLS",
    # 加载函数
    "load_all_tools",
    "load_internal_tools",
    "load_connector_tools",
    "load_skillbook_tools",
    "load_external_tools",
    # 查找函数
    "find_tool",
    "get_tool_names_by_category",
    "get_internal_tool",
    "get_connector_tool",
    "get_skillbook_tool",
    "get_external_tool",
]
