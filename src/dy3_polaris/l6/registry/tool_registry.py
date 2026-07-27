"""MCP 工具注册中心.

Dy3+ Polaris L6 层的核心组件，管理 47 个工具的注册、发现、校验和路由。

核心能力：
1. 工具注册与注销（带 Schema 校验）
2. 多维度发现（名称/分类/层级/标签/领域）
3. 版本管理与更新追踪
4. 工具依赖解析
5. MCP 兼容的工具列表导出
6. 线程安全（asyncio Lock）

架构设计:
    ToolRegistry (中心注册表)
    ├── _tools: dict[str, ToolEntry]          # 主存储
    ├── _category_index: dict[ToolCategory, set[str]]  # 分类索引
    ├── _layer_index: dict[LayerTag, set[str]]         # 层级索引
    ├── _tag_index: dict[str, set[str]]                # 标签倒排索引
    ├── _domain_index: dict[str, set[str]]             # 领域倒排索引
    └── _dependencies: dict[str, list[str]]            # 依赖图
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from ..core.exceptions import L6Error, MCPToolNotFoundError, SchemaValidationError
from ..core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)
from .schema_validator import SchemaValidator, get_validator

logger = logging.getLogger(__name__)


# ============================================================
# 工具条目
# ============================================================

class ToolEntry:
    """注册中心中的工具条目.

    封装 ToolRegistration + handler + 元数据。
    """

    def __init__(
        self,
        registration: ToolRegistration,
        handler: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
        self.registration = registration
        self.handler = handler
        self.dependencies = dependencies or []
        self.registered_at: float = time.time()
        self.updated_at: float = self.registered_at
        self.call_count: int = 0
        self.error_count: int = 0
        self.last_called_at: float | None = None

    @property
    def name(self) -> str:
        return self.registration.name

    @property
    def annotations(self) -> Dy3ToolAnnotations:
        return self.registration.annotations

    @property
    def is_stub(self) -> bool:
        """是否为 stub（无实际 handler）."""
        return self.handler is None

    def touch(self, success: bool = True) -> None:
        """记录一次调用."""
        self.call_count += 1
        if not success:
            self.error_count += 1
        self.last_called_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "name": self.name,
            "description": self.registration.description,
            "version": self.registration.version,
            "enabled": self.registration.enabled,
            "category": self.annotations.category.value,
            "layer": self.annotations.layer.value if self.annotations.layer else None,
            "estimated_latency_ms": self.annotations.estimated_latency_ms,
            "domain_scope": self.annotations.domain_scope,
            "tags": self.annotations.tags,
            "rate_limit": self.annotations.rate_limit,
            "requires_compute": self.annotations.requires_compute,
            "is_stub": self.is_stub,
            "dependencies": self.dependencies,
            "registered_at": self.registered_at,
            "call_count": self.call_count,
            "error_count": self.error_count,
        }


# ============================================================
# 工具注册中心
# ============================================================

class ToolRegistry:
    """MCP 工具注册中心.

    管理所有 MCP 工具的注册、发现和校验。

    使用示例:
        registry = ToolRegistry()

        # 注册工具
        registry.register(my_tool_registration, handler=my_handler)

        # 发现工具
        tools = registry.discover_by_category(ToolCategory.INTERNAL)
        tools = registry.discover_by_layer(LayerTag.L2_PERSONALIZATION)
        tools = registry.discover_by_tag("bkt")
        tool = registry.get("internal.bkt_compute")

        # 导出 MCP 兼容列表
        mcp_tools = registry.export_mcp_tool_list()
    """

    def __init__(
        self,
        *,
        validator: SchemaValidator | None = None,
        auto_validate: bool = True,
    ) -> None:
        self._validator = validator or get_validator()
        self._auto_validate = auto_validate

        # 主存储
        self._tools: dict[str, ToolEntry] = {}

        # 多维索引
        self._category_index: dict[ToolCategory, set[str]] = defaultdict(set)
        self._layer_index: dict[LayerTag, set[str]] = defaultdict(set)
        self._tag_index: dict[str, set[str]] = defaultdict(set)
        self._domain_index: dict[str, set[str]] = defaultdict(set)

        # 依赖图
        self._dependency_graph: dict[str, list[str]] = {}

        # 并发控制
        self._lock = asyncio.Lock()

        # 统计
        self._total_registrations: int = 0
        self._total_updates: int = 0
        self._total_unregistrations: int = 0

    # ============================================================
    # 注册 / 更新 / 注销
    # ============================================================

    async def register(
        self,
        registration: ToolRegistration,
        handler: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> ToolEntry:
        """注册一个工具.

        Args:
            registration: 工具注册信息
            handler: 工具执行函数（None 表示 stub）
            dependencies: 依赖的其他工具名列表
            overwrite: 是否覆盖已存在的同名工具

        Returns:
            创建的 ToolEntry

        Raises:
            L6Error: 工具已存在且 overwrite=False
            SchemaValidationError: Schema 校验失败
        """
        async with self._lock:
            name = registration.name

            if name in self._tools and not overwrite:
                raise L6Error(
                    "TOOL_ALREADY_REGISTERED",
                    f"Tool '{name}' is already registered. Use overwrite=True to replace.",
                    {"tool_name": name},
                )

            # Schema 校验
            if self._auto_validate:
                errors = self._validator.validate_definition(
                    name=registration.name,
                    description=registration.description,
                    input_schema=registration.input_schema,
                    output_schema=registration.output_schema,
                )
                if errors:
                    error_msgs = [f"{e.path}: {e.message}" for e in errors]
                    raise SchemaValidationError(
                        path=name,
                        message=f"Schema validation failed: {'; '.join(error_msgs)}",
                        context={"errors": [e.to_json_rpc_error() for e in errors]},
                    )

            # 创建条目
            entry = ToolEntry(registration, handler, dependencies)

            # 如果覆盖，先清理旧索引
            if name in self._tools:
                self._remove_from_indices(name)
                self._total_updates += 1
            else:
                self._total_registrations += 1

            # 存储
            self._tools[name] = entry

            # 更新索引
            self._add_to_indices(entry)

            # 更新依赖图
            if dependencies:
                self._dependency_graph[name] = dependencies
            else:
                self._dependency_graph.pop(name, None)

            logger.info(
                f"Registered tool: {name} "
                f"[{registration.annotations.category.value}] "
                f"layer={registration.annotations.layer} "
                f"stub={'yes' if handler is None else 'no'}"
            )

            return entry

    async def update(
        self,
        name: str,
        *,
        handler: Callable[..., Any] | None = None,
        registration: ToolRegistration | None = None,
        dependencies: list[str] | None = None,
    ) -> ToolEntry:
        """更新已注册工具.

        Args:
            name: 工具名
            handler: 新的 handler（None 不更新）
            registration: 新的注册信息（None 不更新）
            dependencies: 新的依赖列表（None 不更新）

        Raises:
            MCPToolNotFoundError: 工具不存在
        """
        async with self._lock:
            if name not in self._tools:
                raise MCPToolNotFoundError(name)

            entry = self._tools[name]

            if registration is not None:
                # 重新校验
                if self._auto_validate:
                    errors = self._validator.validate_definition(
                        name=registration.name,
                        description=registration.description,
                        input_schema=registration.input_schema,
                        output_schema=registration.output_schema,
                    )
                    if errors:
                        raise SchemaValidationError(
                            path=registration.name,
                            message=f"Schema validation failed on update",
                            context={"errors": [e.to_json_rpc_error() for e in errors]},
                        )

                # 清理旧索引，添加新索引
                self._remove_from_indices(name)
                entry.registration = registration
                self._add_to_indices(entry)

            if handler is not None:
                entry.handler = handler

            if dependencies is not None:
                self._dependency_graph[name] = dependencies
                entry.dependencies = dependencies

            entry.updated_at = time.time()
            self._total_updates += 1

            logger.info(f"Updated tool: {name}")
            return entry

    async def unregister(self, name: str) -> ToolEntry:
        """注销一个工具.

        Raises:
            MCPToolNotFoundError: 工具不存在
        """
        async with self._lock:
            if name not in self._tools:
                raise MCPToolNotFoundError(name)

            entry = self._tools.pop(name)
            self._remove_from_indices(name)
            self._dependency_graph.pop(name, None)
            self._total_unregistrations += 1

            logger.info(f"Unregistered tool: {name}")
            return entry

    def register_sync(
        self,
        registration: ToolRegistration,
        handler: Callable[..., Any] | None = None,
        dependencies: list[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> ToolEntry:
        """同步注册（用于不方便使用 async 的场景，如模块加载时）."""
        # 复用 async register 的逻辑，但不加锁
        name = registration.name

        if name in self._tools and not overwrite:
            raise L6Error(
                "TOOL_ALREADY_REGISTERED",
                f"Tool '{name}' is already registered. Use overwrite=True to replace.",
                {"tool_name": name},
            )

        if self._auto_validate:
            errors = self._validator.validate_definition(
                name=registration.name,
                description=registration.description,
                input_schema=registration.input_schema,
                output_schema=registration.output_schema,
            )
            if errors:
                error_msgs = [f"{e.path}: {e.message}" for e in errors]
                raise SchemaValidationError(
                    path=name,
                    message=f"Schema validation failed: {'; '.join(error_msgs)}",
                    context={"errors": [e.to_json_rpc_error() for e in errors]},
                )

        entry = ToolEntry(registration, handler, dependencies)

        if name in self._tools:
            self._remove_from_indices(name)
            self._total_updates += 1
        else:
            self._total_registrations += 1

        self._tools[name] = entry
        self._add_to_indices(entry)

        if dependencies:
            self._dependency_graph[name] = dependencies
        else:
            self._dependency_graph.pop(name, None)

        logger.debug(f"Registered (sync) tool: {name}")
        return entry

    def register_batch_sync(
        self,
        entries: list[tuple[ToolRegistration, Callable[..., Any] | None]],
    ) -> list[ToolEntry]:
        """批量同步注册.

        Args:
            entries: [(registration, handler), ...]

        Returns:
            创建的 ToolEntry 列表
        """
        results: list[ToolEntry] = []
        for reg, handler in entries:
            try:
                entry = self.register_sync(reg, handler)
                results.append(entry)
            except Exception as exc:
                logger.error(f"Failed to register tool '{reg.name}': {exc}")
                raise
        return results

    # ============================================================
    # 查询 / 发现
    # ============================================================

    def get(self, name: str) -> ToolEntry | None:
        """按名称获取工具."""
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> ToolEntry:
        """按名称获取工具，不存在时抛出异常."""
        entry = self._tools.get(name)
        if entry is None:
            raise MCPToolNotFoundError(name)
        return entry

    def contains(self, name: str) -> bool:
        """检查工具是否已注册."""
        return name in self._tools

    def discover_by_category(self, category: ToolCategory) -> list[ToolEntry]:
        """按分类发现工具."""
        names = self._category_index.get(category, set())
        return [self._tools[n] for n in names if n in self._tools]

    def discover_by_layer(self, layer: LayerTag) -> list[ToolEntry]:
        """按架构层发现工具."""
        names = self._layer_index.get(layer, set())
        return [self._tools[n] for n in names if n in self._tools]

    def discover_by_tag(self, tag: str) -> list[ToolEntry]:
        """按标签发现工具."""
        names = self._tag_index.get(tag, set())
        return [self._tools[n] for n in names if n in self._tools]

    def discover_by_domain(self, domain: str) -> list[ToolEntry]:
        """按领域发现工具."""
        names = self._domain_index.get(domain, set())
        return [self._tools[n] for n in names if n in self._tools]

    def discover(
        self,
        *,
        category: ToolCategory | None = None,
        layer: LayerTag | None = None,
        tag: str | None = None,
        domain: str | None = None,
        enabled_only: bool = True,
    ) -> list[ToolEntry]:
        """多维度组合发现工具.

        所有非 None 的条件取交集。
        """
        result_sets: list[set[str]] = []

        if category is not None:
            result_sets.append(self._category_index.get(category, set()).copy())
        if layer is not None:
            result_sets.append(self._layer_index.get(layer, set()).copy())
        if tag is not None:
            result_sets.append(self._tag_index.get(tag, set()).copy())
        if domain is not None:
            result_sets.append(self._domain_index.get(domain, set()).copy())

        if not result_sets:
            # 无过滤条件，返回全部
            names = set(self._tools.keys())
        else:
            names = result_sets[0]
            for s in result_sets[1:]:
                names = names & s

        entries = [self._tools[n] for n in names if n in self._tools]
        if enabled_only:
            entries = [e for e in entries if e.registration.enabled]
        return entries

    def search(self, query: str) -> list[ToolEntry]:
        """模糊搜索工具（匹配名称或描述）."""
        query_lower = query.lower()
        results: list[ToolEntry] = []
        for entry in self._tools.values():
            if query_lower in entry.name.lower() or query_lower in entry.registration.description.lower():
                results.append(entry)
        return results

    # ============================================================
    # 依赖解析
    # ============================================================

    def get_dependencies(self, name: str) -> list[str]:
        """获取工具的直接依赖."""
        return list(self._dependency_graph.get(name, []))

    def get_dependents(self, name: str) -> list[str]:
        """获取依赖此工具的其他工具."""
        dependents: list[str] = []
        for tool_name, deps in self._dependency_graph.items():
            if name in deps:
                dependents.append(tool_name)
        return dependents

    def resolve_dependency_chain(self, name: str) -> list[str]:
        """解析完整依赖链（拓扑排序）.

        返回从最底层依赖到目标工具的有序列表。
        检测循环依赖并抛出异常。
        """
        visited: set[str] = set()
        in_stack: set[str] = set()
        result: list[str] = []

        def _visit(node: str) -> None:
            if node in in_stack:
                raise L6Error(
                    "CIRCULAR_DEPENDENCY",
                    f"Circular dependency detected involving '{node}'",
                    {"node": node, "stack": list(in_stack)},
                )
            if node in visited:
                return

            in_stack.add(node)
            for dep in self._dependency_graph.get(node, []):
                if dep in self._tools:
                    _visit(dep)
            in_stack.discard(node)
            visited.add(node)
            result.append(node)

        _visit(name)
        return result

    # ============================================================
    # 导出
    # ============================================================

    def export_mcp_tool_list(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """导出 MCP 兼容的工具列表.

        格式符合 MCP SDK tools/list 响应。
        """
        tools: list[dict[str, Any]] = []
        for entry in self._tools.values():
            if enabled_only and not entry.registration.enabled:
                continue
            tool_dict: dict[str, Any] = {
                "name": entry.name,
                "description": entry.registration.description,
                "inputSchema": entry.registration.input_schema,
            }
            if entry.registration.output_schema:
                tool_dict["outputSchema"] = entry.registration.output_schema
            tools.append(tool_dict)
        return tools

    def export_registry_summary(self) -> dict[str, Any]:
        """导出注册中心摘要统计."""
        category_counts: dict[str, int] = defaultdict(int)
        layer_counts: dict[str, int] = defaultdict(int)
        stub_count = 0
        total_latency = 0

        for entry in self._tools.values():
            category_counts[entry.annotations.category.value] += 1
            if entry.annotations.layer:
                layer_counts[entry.annotations.layer.value] += 1
            if entry.is_stub:
                stub_count += 1
            total_latency += entry.annotations.estimated_latency_ms

        count = len(self._tools)
        return {
            "total_tools": count,
            "enabled_tools": sum(1 for e in self._tools.values() if e.registration.enabled),
            "stub_tools": stub_count,
            "tools_with_handler": count - stub_count,
            "category_breakdown": dict(category_counts),
            "layer_breakdown": dict(layer_counts),
            "avg_estimated_latency_ms": total_latency / count if count > 0 else 0,
            "total_registrations": self._total_registrations,
            "total_updates": self._total_updates,
            "total_unregistrations": self._total_unregistrations,
            "dependency_edges": sum(len(deps) for deps in self._dependency_graph.values()),
        }

    def export_all_entries(self) -> list[dict[str, Any]]:
        """导出所有工具条目的详细信息."""
        return [entry.to_dict() for entry in self._tools.values()]

    # ============================================================
    # 调用统计
    # ============================================================

    def record_call(self, name: str, success: bool = True) -> None:
        """记录一次工具调用结果."""
        entry = self._tools.get(name)
        if entry:
            entry.touch(success)

    def get_call_stats(self, name: str) -> dict[str, Any] | None:
        """获取工具调用统计."""
        entry = self._tools.get(name)
        if entry is None:
            return None
        return {
            "call_count": entry.call_count,
            "error_count": entry.error_count,
            "error_rate": entry.error_count / entry.call_count if entry.call_count > 0 else 0.0,
            "last_called_at": entry.last_called_at,
        }

    # ============================================================
    # 属性
    # ============================================================

    @property
    def size(self) -> int:
        """已注册工具数量."""
        return len(self._tools)

    @property
    def all_names(self) -> list[str]:
        """所有已注册工具名."""
        return list(self._tools.keys())

    @property
    def categories(self) -> list[ToolCategory]:
        """有工具的所有分类."""
        return list(self._category_index.keys())

    # ============================================================
    # 内部方法：索引管理
    # ============================================================

    def _add_to_indices(self, entry: ToolEntry) -> None:
        name = entry.name
        ann = entry.annotations

        self._category_index[ann.category].add(name)

        if ann.layer is not None:
            self._layer_index[ann.layer].add(name)

        for tag in ann.tags:
            self._tag_index[tag].add(name)

        for domain in ann.domain_scope:
            self._domain_index[domain].add(name)

    def _remove_from_indices(self, name: str) -> None:
        entry = self._tools.get(name)
        if entry is None:
            return

        ann = entry.annotations

        self._category_index[ann.category].discard(name)
        if not self._category_index[ann.category]:
            del self._category_index[ann.category]

        if ann.layer is not None:
            self._layer_index[ann.layer].discard(name)
            if not self._layer_index[ann.layer]:
                del self._layer_index[ann.layer]

        for tag in ann.tags:
            self._tag_index[tag].discard(name)
            if not self._tag_index[tag]:
                del self._tag_index[tag]

        for domain in ann.domain_scope:
            self._domain_index[domain].discard(name)
            if not self._domain_index[domain]:
                del self._domain_index[domain]

    def clear(self) -> None:
        """清空注册中心（仅用于测试）."""
        self._tools.clear()
        self._category_index.clear()
        self._layer_index.clear()
        self._tag_index.clear()
        self._domain_index.clear()
        self._dependency_graph.clear()
        self._total_registrations = 0
        self._total_updates = 0
        self._total_unregistrations = 0


# ============================================================
# 全局注册中心单例
# ============================================================

_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """获取全局 ToolRegistry 单例."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """重置全局注册中心（仅用于测试）."""
    global _global_registry
    _global_registry = None
