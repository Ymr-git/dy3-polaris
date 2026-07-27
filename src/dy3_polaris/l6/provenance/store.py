"""溯源存储与查询引擎.

提供多链存储、索引和查询能力：
- 按 chain_id 管理多条链
- 按 kpa_id / actor / event_type / layer / 时间范围 多维查询
- 跨链 KPA 搜索
- 链合并与分支

所有查询均返回 KPA 副本或序列化字典，不暴露内部引用。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.exceptions import KPANotFoundError
from ..core.models import KPA, KPAEventType, LayerTag
from .chain import KPAChain

logger = logging.getLogger(__name__)


class ProvenanceStore:
    """溯源存储引擎.

    管理多条 KPA 链并提供多维查询接口。
    当前为内存存储，生产环境可替换为持久化后端。

    使用示例:
        store = ProvenanceStore()
        chain = store.create_chain("session-001")
        chain.append(KPAEventType.TOOL_INVOKED, "bkt_compute", LayerTag.L6_PROTOCOL)
        # 查询
        results = store.query_by_actor("bkt_compute")
    """

    def __init__(self) -> None:
        self._chains: dict[str, KPAChain] = {}
        # 全局 KPA 索引: kpa_id -> (chain_id, index)
        self._kpa_index: dict[str, tuple[str, int]] = {}

    # --------------------------------------------------------
    # 链管理
    # --------------------------------------------------------

    def create_chain(self, chain_id: str = "") -> KPAChain:
        """创建新链.

        Args:
            chain_id: 链 ID，若为空则自动生成

        Returns:
            新创建的 KPAChain
        """
        if not chain_id:
            chain_id = f"chain-{len(self._chains):04d}"
        if chain_id in self._chains:
            logger.warning("链 %s 已存在，返回已有链", chain_id)
            return self._chains[chain_id]

        chain = KPAChain(chain_id=chain_id)
        self._chains[chain_id] = chain
        logger.debug("创建链: %s", chain_id)
        return chain

    def get_chain(self, chain_id: str) -> KPAChain | None:
        """获取链."""
        return self._chains.get(chain_id)

    def get_chain_or_raise(self, chain_id: str) -> KPAChain:
        """获取链，不存在则抛异常."""
        chain = self._chains.get(chain_id)
        if chain is None:
            raise KPANotFoundError(f"chain:{chain_id}")
        return chain

    def remove_chain(self, chain_id: str) -> bool:
        """移除链及其索引."""
        chain = self._chains.pop(chain_id, None)
        if chain is None:
            return False
        # 清理 KPA 索引
        kpa_ids_to_remove = [kid for kid, (cid, _) in self._kpa_index.items() if cid == chain_id]
        for kid in kpa_ids_to_remove:
            del self._kpa_index[kid]
        return True

    @property
    def chain_count(self) -> int:
        return len(self._chains)

    @property
    def total_kpa_count(self) -> int:
        return sum(chain.length for chain in self._chains.values())

    def all_chain_ids(self) -> list[str]:
        return list(self._chains.keys())

    # --------------------------------------------------------
    # KPA 索引维护
    # --------------------------------------------------------

    def _rebuild_index(self) -> None:
        """重建全局 KPA 索引."""
        self._kpa_index.clear()
        for chain_id, chain in self._chains.items():
            for i, kpa in enumerate(chain.kpas):
                self._kpa_index[kpa.kpa_id] = (chain_id, i)

    def _index_chain(self, chain_id: str, chain: KPAChain) -> None:
        """为单条链建立索引."""
        for i, kpa in enumerate(chain.kpas):
            self._kpa_index[kpa.kpa_id] = (chain_id, i)

    # --------------------------------------------------------
    # KPA 查询
    # --------------------------------------------------------

    def get_kpa(self, kpa_id: str) -> KPA | None:
        """按 kpa_id 全局查找 KPA."""
        # 先检查索引是否需要更新
        location = self._kpa_index.get(kpa_id)
        if location is None:
            # 索引未命中，重建后重试
            self._rebuild_index()
            location = self._kpa_index.get(kpa_id)
            if location is None:
                return None

        chain_id, idx = location
        chain = self._chains.get(chain_id)
        if chain is None:
            return None
        return chain.get(idx)

    def get_kpa_location(self, kpa_id: str) -> tuple[str, int] | None:
        """获取 KPA 所在的链 ID 和索引."""
        if kpa_id not in self._kpa_index:
            self._rebuild_index()
        return self._kpa_index.get(kpa_id)

    def query(
        self,
        *,
        actor: str | None = None,
        event_type: KPAEventType | None = None,
        layer: LayerTag | None = None,
        chain_id: str | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[KPA]:
        """多条件查询 KPA.

        所有条件均为 AND 关系，None 表示不限制该维度。

        Args:
            actor: 按执行者筛选
            event_type: 按事件类型筛选
            layer: 按层标签筛选
            chain_id: 限定链 ID
            time_from: 起始时间戳（含）
            time_to: 结束时间戳（含）
            min_confidence: 最低置信度
            limit: 最多返回条数

        Returns:
            匹配的 KPA 列表
        """
        results: list[KPA] = []

        # 确定搜索范围
        if chain_id is not None:
            chains = [self._chains.get(chain_id)] if chain_id in self._chains else []
        else:
            chains = list(self._chains.values())

        for chain in chains:
            if chain is None:
                continue
            for kpa in chain.kpas:
                # 筛选条件
                if actor is not None and kpa.actor != actor:
                    continue
                if event_type is not None and kpa.event_type != event_type:
                    continue
                if layer is not None and kpa.layer != layer:
                    continue
                if time_from is not None and kpa.timestamp < time_from:
                    continue
                if time_to is not None and kpa.timestamp > time_to:
                    continue
                if min_confidence is not None:
                    if kpa.confidence is None or kpa.confidence < min_confidence:
                        continue
                results.append(kpa)
                if len(results) >= limit:
                    return results

        return results

    def query_by_actor(self, actor: str, *, chain_id: str | None = None) -> list[KPA]:
        """按执行者查询."""
        return self.query(actor=actor, chain_id=chain_id)

    def query_by_event_type(self, event_type: KPAEventType, *, chain_id: str | None = None) -> list[KPA]:
        """按事件类型查询."""
        return self.query(event_type=event_type, chain_id=chain_id)

    def query_by_layer(self, layer: LayerTag, *, chain_id: str | None = None) -> list[KPA]:
        """按层标签查询."""
        return self.query(layer=layer, chain_id=chain_id)

    def query_by_time_range(self, time_from: float, time_to: float, *, chain_id: str | None = None) -> list[KPA]:
        """按时间范围查询."""
        return self.query(time_from=time_from, time_to=time_to, chain_id=chain_id)

    def query_low_confidence(self, threshold: float = 0.5, *, chain_id: str | None = None) -> list[KPA]:
        """查询低置信度 KPA（低于阈值且非 None）."""
        results: list[KPA] = []
        chains = [self._chains.get(chain_id)] if chain_id else list(self._chains.values())
        for chain in chains:
            if chain is None:
                continue
            for kpa in chain.kpas:
                if kpa.confidence is not None and kpa.confidence < threshold:
                    results.append(kpa)
        return results

    # --------------------------------------------------------
    # 跨链搜索
    # --------------------------------------------------------

    def find_all_actors(self) -> list[str]:
        """获取所有出现过的执行者."""
        actors: set[str] = set()
        for chain in self._chains.values():
            for kpa in chain.kpas:
                actors.add(kpa.actor)
        return sorted(actors)

    def find_all_event_types(self) -> list[str]:
        """获取所有出现过的事件类型."""
        types: set[str] = set()
        for chain in self._chains.values():
            for kpa in chain.kpas:
                types.add(kpa.event_type.value)
        return sorted(types)

    def find_context_refs(self, ref_pattern: str = "") -> list[tuple[str, str]]:
        """搜索上下文引用.

        Args:
            ref_pattern: 引用模式（子串匹配），空则返回全部

        Returns:
            (kpa_id, ref) 列表
        """
        results: list[tuple[str, str]] = []
        for chain in self._chains.values():
            for kpa in chain.kpas:
                for ref in kpa.context_refs:
                    if not ref_pattern or ref_pattern in ref:
                        results.append((kpa.kpa_id, ref))
        return results

    # --------------------------------------------------------
    # 统计与导出
    # --------------------------------------------------------

    def chain_summary(self, chain_id: str) -> dict[str, Any] | None:
        """获取单条链的摘要."""
        chain = self._chains.get(chain_id)
        if chain is None:
            return None
        return {
            "chain_id": chain_id,
            "length": chain.length,
            "sealed": chain.is_sealed,
            "head_hash": chain.head_hash,
            "event_type_counts": chain.event_type_counts(),
            "actor_counts": chain.actor_counts(),
            "layer_counts": chain.layer_counts(),
            "avg_confidence": chain.avg_confidence(),
            "duration_seconds": chain.duration_seconds(),
        }

    def export_all(self) -> dict[str, Any]:
        """导出所有链的完整数据."""
        return {
            "chain_count": len(self._chains),
            "total_kpas": self.total_kpa_count,
            "all_actors": self.find_all_actors(),
            "all_event_types": self.find_all_event_types(),
            "chains": {cid: chain.to_dict() for cid, chain in self._chains.items()},
        }

    def export_summary(self) -> dict[str, Any]:
        """导出摘要（不含 KPA 明细）."""
        return {
            "chain_count": len(self._chains),
            "total_kpas": self.total_kpa_count,
            "chains": {cid: self.chain_summary(cid) for cid in self._chains},
        }

    # --------------------------------------------------------
    # 重置
    # --------------------------------------------------------

    def clear(self) -> None:
        """清空所有链和索引."""
        self._chains.clear()
        self._kpa_index.clear()
        logger.debug("溯源存储已清空")


__all__ = ["ProvenanceStore"]
