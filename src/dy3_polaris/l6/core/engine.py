"""L6 核心引擎.

MCP Server 生命周期管理，统一协调各子模块。
提供统一的工具调用入口。
"""

from __future__ import annotations

import logging
from typing import Any

from .config import L6Config
from .exceptions import L6Error, MethodNotFoundError


logger = logging.getLogger("dy3_polaris.l6.core.engine")


class L6CoreEngine:
    """L6 核心引擎.

    管理所有 L6 子模块的生命周期。
    提供统一的工具调用入口。
    """

    def __init__(self, config: L6Config | None = None) -> None:
        self.config = config or L6Config()
        self._logger = logging.getLogger("dy3_polaris.l6.core.engine")

        # 子模块引用
        self.tool_registry = None
        self.a2a_bus = None
        self.compute_scheduler = None
        self.broadcast_bus = None
        self.memory_graph = None
        self.provenance_store = None

    def initialize(self) -> None:
        """按依赖顺序初始化所有子模块."""
        self._logger.info("L6 核心引擎初始化开始")

        # T2: 工具注册中心
        from dy3_polaris.l6.registry.tool_registry import ToolRegistry
        self.tool_registry = ToolRegistry()
        self._logger.info("工具注册中心已初始化")

        # T3: A2A 协议
        if self.config.a2a_enabled:
            from dy3_polaris.l6.a2a.protocol import A2AMessageBus
            self.a2a_bus = A2AMessageBus()
            self._logger.info("A2A 消息总线已初始化")

        # T5: 算力调度
        from dy3_polaris.l6.compute.scheduler import ComputeScheduler
        self.compute_scheduler = ComputeScheduler()
        self._logger.info("算力调度器已初始化")

        # T6: 广播总线
        if self.config.broadcast_enabled:
            from dy3_polaris.l6.broadcast.broadcast import BroadcastBus
            self.broadcast_bus = BroadcastBus(
                max_subscribers_per_topic=self.config.broadcast_max_subscribers_per_topic,
                event_log_enabled=self.config.broadcast_event_log_enabled,
                event_log_max_size=self.config.broadcast_event_log_max_size,
            )
            self._logger.info("广播总线已初始化")

        # T6: 记忆图谱
        if self.config.memory_graph_enabled:
            from dy3_polaris.l6.broadcast.memory_graph import MemoryGraph
            self.memory_graph = MemoryGraph(
                decay_factor=self.config.memory_graph_decay_factor,
                min_strength=self.config.memory_graph_min_strength,
                spreading_depth=self.config.memory_graph_spreading_depth,
                spreading_decay=self.config.memory_graph_spreading_decay,
            )
            self._logger.info("记忆图谱已初始化")

        # T4: 溯源存储
        if self.config.provenance_enabled:
            from dy3_polaris.l6.provenance.store import ProvenanceStore
            self.provenance_store = ProvenanceStore()
            self._logger.info("溯源存储已初始化")

        self._logger.info("L6 核心引擎初始化完成")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """统一工具调用入口.

        Args:
            tool_name: 工具名
            arguments: 工具参数

        Returns:
            {"result": ..., "kpa_id": ...} 或 {"error": ...}
        """
        reg = self.tool_registry
        if reg is None:
            return {"error": {"code": -32000, "message": "工具注册中心未初始化"}}

        entry = reg.get(tool_name)
        if entry is None:
            return {"error": MethodNotFoundError(tool_name).to_json_rpc_error()}

        if entry.handler is None:
            return {"error": {"code": -32000, "message": f"工具 {tool_name!r} 无可用 handler (stub)"}}

        try:
            result = entry.handler(**arguments)
            entry.touch(success=True)
            kpa_id = self._record_provenance(
                tool_name, arguments, result if isinstance(result, dict) else {"value": result},
            )
            return {"result": result, "kpa_id": kpa_id}
        except Exception as e:
            entry.touch(success=False)
            self._record_provenance(
                tool_name, arguments, {"error": str(e)},
            )
            return {"error": {"code": -32000, "message": str(e)}}

    def _record_provenance(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output: dict[str, Any],
    ) -> str | None:
        """记录工具调用到溯源链, 返回 kpa_id."""
        store = self.provenance_store
        if store is None:
            return None
        try:
            from dy3_polaris.l6.core.models import KPAEventType, LayerTag
            chain = store.create_chain()
            kpa = chain.append(
                event_type=KPAEventType.TOOL_INVOKED,
                actor="l6-engine",
                layer=LayerTag.L6_PROTOCOL,
                input_snapshot=arguments,
                output_snapshot=output,
            )
            return kpa.kpa_id
        except Exception:
            return None

    def shutdown(self) -> None:
        """优雅关闭."""
        self._logger.info("L6 核心引擎关闭")
        self.tool_registry = None
        self.a2a_bus = None
        self.compute_scheduler = None
        self.broadcast_bus = None
        self.memory_graph = None
        self.provenance_store = None

    def get_status(self) -> dict[str, Any]:
        """获取引擎状态."""
        return {
            "initialized": self.tool_registry is not None,
            "modules": {
                "tool_registry": self.tool_registry is not None,
                "a2a_bus": self.config.a2a_enabled and self.a2a_bus is not None,
                "compute_scheduler": self.compute_scheduler is not None,
                "broadcast_bus": self.config.broadcast_enabled and self.broadcast_bus is not None,
                "memory_graph": self.config.memory_graph_enabled and self.memory_graph is not None,
                "provenance_store": self.config.provenance_enabled and self.provenance_store is not None,
            },
        }
