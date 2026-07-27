"""KPA Merkle 链管理器.

实现溯源链的核心逻辑：
- KPA 追加与链接
- Merkle 哈希计算与验证
- 链分支与合并
- 链快照与回滚

每个 KPA 通过 prev_hash 指向前一个 KPA 的哈希，
形成类似区块链的防篡改链式结构。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from ..core.exceptions import KPAChainBrokenError, KPAImmutableError
from ..core.models import KPA, KPAEventType, LayerTag
from ..core.utils import snapshot_sanitize

logger = logging.getLogger(__name__)


class KPAChain:
    """KPA Merkle 链.

    管理一个有序的 KPA 序列，每个 KPA 的 prev_hash
    指向前一个 KPA 的 compute_hash()，形成防篡改链。

    使用示例:
        chain = KPAChain()
        kpa1 = chain.append(KPAEventType.TOOL_INVOKED, "bkt_compute", LayerTag.L6_PROTOCOL,
                            input_snapshot={"learner_id": "u001"})
        kpa2 = chain.append(KPAEventType.AGENT_OUTPUT, "tutor-agent", LayerTag.L5_AGENT_RUNTIME,
                            input_snapshot={"result": "pass"})
        assert chain.verify() is True
    """

    def __init__(self, chain_id: str = "") -> None:
        """初始化空链.

        Args:
            chain_id: 链 ID（可选，用于标识不同会话的链）
        """
        self._chain_id: str = chain_id
        self._kpas: list[KPA] = []
        self._sealed: bool = False
        self._created_at: float = time.time()

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @property
    def length(self) -> int:
        return len(self._kpas)

    @property
    def is_empty(self) -> bool:
        return len(self._kpas) == 0

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def head_hash(self) -> str | None:
        """链头（最新 KPA）的哈希."""
        if not self._kpas:
            return None
        return self._kpas[-1].compute_hash()

    @property
    def genesis_hash(self) -> str | None:
        """创世 KPA（第一个）的哈希."""
        if not self._kpas:
            return None
        return self._kpas[0].compute_hash()

    @property
    def kpas(self) -> list[KPA]:
        """返回 KPA 列表的副本."""
        return list(self._kpas)

    # --------------------------------------------------------
    # KPA 追加
    # --------------------------------------------------------

    def append(
        self,
        event_type: KPAEventType,
        actor: str,
        layer: LayerTag,
        *,
        input_snapshot: dict[str, Any] | None = None,
        processing_logic: str = "",
        output_snapshot: dict[str, Any] | None = None,
        context_refs: list[str] | None = None,
        confidence: float | None = None,
        code_hash: str | None = None,
        env_hash: str | None = None,
    ) -> KPA:
        """追加一个 KPA 到链尾.

        自动设置 prev_hash 为当前链尾 KPA 的哈希。

        Args:
            event_type: 事件类型
            actor: 执行者标识
            layer: 执行层标签
            input_snapshot: 输入快照
            processing_logic: 处理逻辑标识
            output_snapshot: 输出快照
            context_refs: 上下文引用列表
            confidence: 置信度 [0,1]
            code_hash: 代码 Git commit hash
            env_hash: 环境配置 hash

        Returns:
            新创建的 KPA

        Raises:
            KPAImmutableError: 链已封存
        """
        if self._sealed:
            raise KPAImmutableError("chain", {"chain_id": self._chain_id})

        prev_hash = self._kpas[-1].compute_hash() if self._kpas else None

        kpa = KPA(
            prev_hash=prev_hash,
            event_type=event_type,
            actor=actor,
            layer=layer,
            input_snapshot=snapshot_sanitize(input_snapshot or {}),
            processing_logic=processing_logic,
            output_snapshot=snapshot_sanitize(output_snapshot or {}),
            context_refs=context_refs or [],
            confidence=confidence,
            code_hash=code_hash,
            env_hash=env_hash,
        )
        self._kpas.append(kpa)

        logger.debug(
            "KPA 已追加到链 %s: idx=%d, kpa_id=%s, event=%s, actor=%s",
            self._chain_id or "<default>",
            len(self._kpas) - 1,
            kpa.kpa_id,
            event_type.value,
            actor,
        )
        return kpa

    def append_existing(self, kpa: KPA) -> KPA:
        """追加一个已构造的 KPA 到链尾.

        会自动修正 prev_hash 以保持链连续性。

        Args:
            kpa: 要追加的 KPA

        Returns:
            追加后的 KPA（prev_hash 可能被修正）

        Raises:
            KPAImmutableError: 链已封存
        """
        if self._sealed:
            raise KPAImmutableError("chain", {"chain_id": self._chain_id})

        prev_hash = self._kpas[-1].compute_hash() if self._kpas else None
        kpa.prev_hash = prev_hash
        self._kpas.append(kpa)
        return kpa

    # --------------------------------------------------------
    # 链操作
    # --------------------------------------------------------

    def seal(self) -> None:
        """封存链，禁止后续追加.

        封存后的链不可修改，适用于会话结束后的溯源固化。
        """
        self._sealed = True
        logger.info("链 %s 已封存，长度=%d", self._chain_id or "<default>", len(self._kpas))

    def unseal(self) -> None:
        """解除封存（仅用于测试）."""
        self._sealed = False

    def get(self, index: int) -> KPA | None:
        """按索引获取 KPA."""
        if 0 <= index < len(self._kpas):
            return self._kpas[index]
        return None

    def get_by_id(self, kpa_id: str) -> KPA | None:
        """按 kpa_id 查找 KPA."""
        for kpa in self._kpas:
            if kpa.kpa_id == kpa_id:
                return kpa
        return None

    def slice(self, start: int = 0, end: int | None = None) -> list[KPA]:
        """获取链的子段."""
        return self._kpas[start:end] if end else self._kpas[start:]

    def snapshot(self) -> list[dict[str, Any]]:
        """获取链快照（序列化）."""
        return [kpa.model_dump(mode="json") for kpa in self._kpas]

    def rollback(self, to_index: int) -> int:
        """回滚链到指定索引.

        移除索引之后的所有 KPA。

        Args:
            to_index: 保留到该索引（含）

        Returns:
            被移除的 KPA 数量

        Raises:
            KPAImmutableError: 链已封存
            IndexError: 索引越界
        """
        if self._sealed:
            raise KPAImmutableError("chain", {"chain_id": self._chain_id})
        if to_index < 0 or to_index >= len(self._kpas):
            raise IndexError(f"回滚索引越界: {to_index}, 链长度={len(self._kpas)}")

        removed = len(self._kpas) - (to_index + 1)
        self._kpas = self._kpas[: to_index + 1]
        logger.info("链 %s 回滚到索引 %d, 移除 %d 个 KPA", self._chain_id, to_index, removed)
        return removed

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def event_type_counts(self) -> dict[str, int]:
        """按事件类型统计."""
        counts: dict[str, int] = {}
        for kpa in self._kpas:
            key = kpa.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def actor_counts(self) -> dict[str, int]:
        """按执行者统计."""
        counts: dict[str, int] = {}
        for kpa in self._kpas:
            counts[kpa.actor] = counts.get(kpa.actor, 0) + 1
        return counts

    def layer_counts(self) -> dict[str, int]:
        """按层标签统计."""
        counts: dict[str, int] = {}
        for kpa in self._kpas:
            key = kpa.layer.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def duration_seconds(self) -> float:
        """链的时间跨度（秒）."""
        if len(self._kpas) < 2:
            return 0.0
        return self._kpas[-1].timestamp - self._kpas[0].timestamp

    def avg_confidence(self) -> float | None:
        """平均置信度."""
        confidences = [kpa.confidence for kpa in self._kpas if kpa.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典."""
        return {
            "chain_id": self._chain_id,
            "length": len(self._kpas),
            "sealed": self._sealed,
            "created_at": self._created_at,
            "head_hash": self.head_hash,
            "genesis_hash": self.genesis_hash,
            "duration_seconds": round(self.duration_seconds(), 3),
            "event_type_counts": self.event_type_counts(),
            "actor_counts": self.actor_counts(),
            "layer_counts": self.layer_counts(),
            "avg_confidence": round(self.avg_confidence(), 4) if self.avg_confidence() is not None else None,
            "kpas": self.snapshot(),
        }

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    # --------------------------------------------------------
    # 重置
    # --------------------------------------------------------

    def clear(self) -> None:
        """清空链."""
        self._kpas.clear()
        self._sealed = False
        self._created_at = time.time()
        logger.debug("链 %s 已清空", self._chain_id)


__all__ = ["KPAChain"]
