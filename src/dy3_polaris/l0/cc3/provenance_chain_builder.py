"""CC3 溯源捕获层 — 溯源链构建器 (Provenance Chain Builder).

实现不可篡改的溯源链式结构:
- SHA-256 哈希链: 每个节点的 prev_hash 指向前一节点的 node_hash
- Merkle 树压缩: 批量节点的高效完整性证明 (RFC 6962)
- 跨层传递: 8 种跨层方向的溯源节点追踪
- 链式验证: 全链路 tamper detection

核心能力:
- 创建溯源链, 追加节点 (append-only)
- 自动计算 prev_hash 与 node_hash
- Merkle 树构建与根哈希计算
- Merkle 证明生成与验证 (稀疏证明)
- 全链完整性校验 (断链检测)
- 跨层传递方向追踪
- 链压缩与快照

融合方案:
- RFC 6962 Certificate Transparency: Merkle 树 append-only 日志
- 区块链: prev_hash 链式校验 (tamper-evident)
- OpenTelemetry: span 父子关系 (trace 上下文传递)
- W3C PROV: wasDerivedFrom / wasGeneratedBy 关系映射
- Apache Iceberg: 快照式版本管理
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    AuditVerificationResult,
    CrossLayerDirection,
    ProvenanceChainNode,
)
from .exceptions import (
    CC3Error,
    ChainBrokenError,
    HashMismatchError,
)

logger = logging.getLogger(__name__)


# ============================================================
# Merkle 树
# ============================================================


class MerkleTree:
    """Merkle 树 — 批量完整性证明.

    将多个节点的哈希组织为二叉 Merkle 树,
    支持高效的包含证明 (inclusion proof)。

    RFC 6962 启发:
    - 叶子节点: SHA-256(节点哈希)
    - 内部节点: SHA-256(左子哈希 || 右子哈希)
    - 奇数叶子: 复制最后一个叶子

    使用示例::

        tree = MerkleTree()
        tree.add_leaf("hash1")
        tree.add_leaf("hash2")
        tree.add_leaf("hash3")
        root = tree.build()
        proof = tree.get_proof(0)  # 第0个叶子的包含证明
        assert tree.verify_proof("hash1", proof, root)
    """

    def __init__(self) -> None:
        """初始化空 Merkle 树."""
        self._leaves: list[str] = []
        self._tree: list[list[str]] = []
        self._root: str = ""

    def add_leaf(self, leaf_hash: str) -> int:
        """添加叶子节点.

        Args:
            leaf_hash: 叶子节点的哈希值

        Returns:
            叶子索引
        """
        self._leaves.append(leaf_hash)
        return len(self._leaves) - 1

    def build(self) -> str:
        """构建 Merkle 树, 返回根哈希.

        Returns:
            Merkle 根哈希 (SHA-256)
        """
        if not self._leaves:
            self._root = hashlib.sha256(b"").hexdigest()
            self._tree = []
            return self._root

        # 叶子层: SHA-256(原始哈希)
        current_level = [
            hashlib.sha256(h.encode("utf-8")).hexdigest()
            for h in self._leaves
        ]
        self._tree = [list(current_level)]

        # 逐层向上构建
        while len(current_level) > 1:
            next_level: list[str] = []
            # 奇数叶子: 复制最后一个
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                parent = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                next_level.append(parent)

            self._tree.append(list(next_level))
            current_level = next_level

        self._root = current_level[0]
        return self._root

    @property
    def root(self) -> str:
        """返回 Merkle 根哈希."""
        return self._root

    @property
    def leaf_count(self) -> int:
        """返回叶子数量."""
        return len(self._leaves)

    def get_proof(self, leaf_index: int) -> list[dict[str, str]]:
        """生成包含证明 (Merkle Proof).

        对于指定叶子, 返回从叶子到根路径上每层的兄弟节点哈希。

        Args:
            leaf_index: 叶子索引

        Returns:
            证明路径::

                [
                    {"hash": str, "direction": "left"|"right"},
                    ...
                ]
        """
        if not self._tree or leaf_index < 0 or leaf_index >= len(self._leaves):
            return []

        proof: list[dict[str, str]] = []
        index = leaf_index

        for level in range(len(self._tree) - 1):
            current_level = self._tree[level]
            # 确定兄弟节点
            if index % 2 == 0:
                # 当前是左子, 兄弟是右子
                sibling_index = index + 1
                direction = "right"
            else:
                # 当前是右子, 兄弟是左子
                sibling_index = index - 1
                direction = "left"

            if sibling_index < len(current_level):
                proof.append({
                    "hash": current_level[sibling_index],
                    "direction": direction,
                })

            index = index // 2

        return proof

    def verify_proof(
        self,
        leaf_hash: str,
        proof: list[dict[str, str]],
        root_hash: str,
    ) -> bool:
        """验证包含证明.

        Args:
            leaf_hash: 叶子哈希
            proof: Merkle 证明路径
            root_hash: 预期的 Merkle 根哈希

        Returns:
            证明是否有效
        """
        current = hashlib.sha256(leaf_hash.encode("utf-8")).hexdigest()

        for step in proof:
            sibling = step["hash"]
            if step["direction"] == "left":
                # 兄弟在左
                combined = sibling + current
            else:
                # 兄弟在右
                combined = current + sibling
            current = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return current == root_hash

    def serialize(self) -> dict[str, Any]:
        """序列化 Merkle 树."""
        return {
            "leaves": list(self._leaves),
            "tree": [list(level) for level in self._tree],
            "root": self._root,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "MerkleTree":
        """反序列化 Merkle 树."""
        tree = cls()
        tree._leaves = list(data.get("leaves", []))
        tree._tree = [list(level) for level in data.get("tree", [])]
        tree._root = data.get("root", "")
        return tree


# ============================================================
# 溯源链构建器
# ============================================================


class ProvenanceChainBuilder:
    """溯源链构建器 — 不可篡改的链式溯源结构.

    管理溯源链的完整生命周期:
    1. create_chain(): 创建新链
    2. append_node(): 追加节点 (自动链接 prev_hash)
    3. verify_chain(): 全链完整性验证
    4. build_merkle_tree(): 构建 Merkle 树
    5. get_merkle_proof(): 获取节点包含证明
    6. compress(): 链压缩 (Merkle 根替代全链)
    7. snapshot(): 创建链快照

    链式结构:
    - Node[0]: prev_hash = "0" * 64 (创世节点)
    - Node[i]: prev_hash = Node[i-1].node_hash
    - 任何节点篡改 → node_hash 变化 → 后续节点 prev_hash 不匹配

    使用示例::

        builder = ProvenanceChainBuilder()
        chain_id = builder.create_chain("chain-001")
        builder.append_node(chain_id, annotation_id="kpa-001", agent_id="agent-1")
        builder.append_node(chain_id, annotation_id="kpa-002", agent_id="agent-2")
        report = builder.verify_chain(chain_id)
        assert report["all_passed"]
    """

    # 创世节点的 prev_hash (64个0)
    GENESIS_PREV_HASH = "0" * 64

    def __init__(self) -> None:
        """初始化溯源链构建器."""
        self._chains: dict[str, list[ProvenanceChainNode]] = {}
        self._merkle_trees: dict[str, MerkleTree] = {}
        self._chain_metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ==========================================================
    # 链创建
    # ==========================================================

    def create_chain(
        self,
        chain_id: str = "",
        description: str = "",
    ) -> str:
        """创建新的溯源链.

        Args:
            chain_id: 链 ID (空则自动生成)
            description: 链描述

        Returns:
            链 ID
        """
        with self._lock:
            cid = chain_id or f"chain-{uuid.uuid4().hex[:10]}"
            if cid in self._chains:
                raise ValueError(f"链已存在: {cid}")
            self._chains[cid] = []
            self._merkle_trees[cid] = MerkleTree()
            self._chain_metadata[cid] = {
                "chain_id": cid,
                "description": description,
                "created_at": time.time(),
                "node_count": 0,
                "root_hash": "",
            }
            logger.info("创建溯源链: id=%s, desc=%s", cid, description)
            return cid

    # ==========================================================
    # 节点追加
    # ==========================================================

    def append_node(
        self,
        chain_id: str,
        annotation_id: str = "",
        target_id: str = "",
        agent_id: str = "",
        agent_role: str = "annotator",
        layer: str = "",
        direction: CrossLayerDirection | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> ProvenanceChainNode:
        """追加溯源节点到链尾.

        自动设置:
        - node_index: 递增序号
        - prev_hash: 前一节点的 node_hash (创世节点为全0)
        - node_hash: 本节点内容哈希

        Args:
            chain_id: 链 ID
            annotation_id: 关联的 KPA 标注 ID
            target_id: 处理对象 ID
            agent_id: 处理 Agent ID
            agent_role: Agent 角色
            layer: 所属架构层
            direction: 跨层传递方向
            extra_data: 额外数据 (写入 payload)

        Returns:
            创建的 ProvenanceChainNode

        Raises:
            ValueError: 链不存在
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            node_index = len(chain)
            prev_hash = (
                chain[-1].node_hash if chain else self.GENESIS_PREV_HASH
            )

            node = ProvenanceChainNode(
                chain_id=chain_id,
                node_index=node_index,
                annotation_id=annotation_id,
                target_id=target_id,
                agent_id=agent_id,
                agent_role=agent_role,
                layer=layer,
                direction=direction,
                timestamp=time.time(),
                prev_hash=prev_hash,
            )
            node.node_hash = node.compute_node_hash()

            chain.append(node)

            # 更新 Merkle 树
            self._merkle_trees[chain_id].add_leaf(node.node_hash)

            # 更新元数据
            self._chain_metadata[chain_id]["node_count"] = len(chain)

            logger.debug(
                "追加溯源节点: chain=%s, index=%d, agent=%s, layer=%s",
                chain_id,
                node_index,
                agent_id,
                layer,
            )
            return node

    # ==========================================================
    # 链验证
    # ==========================================================

    def verify_chain(self, chain_id: str) -> dict[str, Any]:
        """验证整条溯源链的完整性.

        检查:
        - 每个节点的 prev_hash 是否等于前一节点的 node_hash
        - 每个节点的 node_hash 是否等于其内容计算出的哈希
        - 时间戳是否单调递增
        - Agent 一致性 (可选)

        Args:
            chain_id: 链 ID

        Returns:
            验证报告::

                {
                    "chain_id": str,
                    "total_nodes": int,
                    "passed_nodes": int,
                    "failed_nodes": int,
                    "hash_chain_verified": bool,
                    "timestamp_monotonic": bool,
                    "all_passed": bool,
                    "failures": [...],
                }

        Raises:
            ValueError: 链不存在
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            total = len(chain)
            passed = 0
            failed = 0
            failures: list[dict[str, Any]] = []
            hash_chain_verified = True
            timestamp_monotonic = True

            for i, node in enumerate(chain):
                node_failures: list[str] = []

                # 检查 prev_hash
                expected_prev = (
                    chain[i - 1].node_hash if i > 0 else self.GENESIS_PREV_HASH
                )
                if node.prev_hash != expected_prev:
                    hash_chain_verified = False
                    node_failures.append(
                        f"prev_hash 不匹配: expected={expected_prev[:16]}..., actual={node.prev_hash[:16]}..."
                    )

                # 检查 node_hash
                computed_hash = node.compute_node_hash()
                if node.node_hash != computed_hash:
                    hash_chain_verified = False
                    node_failures.append(
                        f"node_hash 不匹配: stored={node.node_hash[:16]}..., computed={computed_hash[:16]}..."
                    )

                # 检查时间戳单调递增
                if i > 0 and node.timestamp < chain[i - 1].timestamp:
                    timestamp_monotonic = False
                    node_failures.append(
                        f"时间戳非单调递增: prev={chain[i-1].timestamp}, current={node.timestamp}"
                    )

                if node_failures:
                    failed += 1
                    failures.append({
                        "node_index": i,
                        "node_id": node.node_id,
                        "issues": node_failures,
                    })
                else:
                    passed += 1

            all_passed = (
                passed == total
                and hash_chain_verified
                and timestamp_monotonic
            )

            return {
                "chain_id": chain_id,
                "total_nodes": total,
                "passed_nodes": passed,
                "failed_nodes": failed,
                "hash_chain_verified": hash_chain_verified,
                "timestamp_monotonic": timestamp_monotonic,
                "all_passed": all_passed,
                "failures": failures,
            }

    def verify_node(self, chain_id: str, node_index: int) -> bool:
        """验证单个节点的完整性.

        Args:
            chain_id: 链 ID
            node_index: 节点序号

        Returns:
            节点是否完整

        Raises:
            ChainBrokenError: prev_hash 不匹配
            HashMismatchError: node_hash 不匹配
            ValueError: 链或节点不存在
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            if node_index < 0 or node_index >= len(chain):
                raise ValueError(f"节点不存在: index={node_index}")

            node = chain[node_index]
            expected_prev = (
                chain[node_index - 1].node_hash
                if node_index > 0
                else self.GENESIS_PREV_HASH
            )

            if node.prev_hash != expected_prev:
                raise ChainBrokenError(
                    chain_id=chain_id,
                    broken_at_index=node_index,
                    expected_prev=expected_prev,
                    actual_prev=node.prev_hash,
                )

            computed = node.compute_node_hash()
            if node.node_hash != computed:
                raise HashMismatchError(
                    expected_hash=node.node_hash,
                    actual_hash=computed,
                    record_id=node.node_id,
                )

            return True

    # ==========================================================
    # Merkle 树操作
    # ==========================================================

    def build_merkle_tree(self, chain_id: str) -> str:
        """构建 (或重建) 链的 Merkle 树.

        Args:
            chain_id: 链 ID

        Returns:
            Merkle 根哈希
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            tree = MerkleTree()
            for node in chain:
                tree.add_leaf(node.node_hash)

            root = tree.build()
            self._merkle_trees[chain_id] = tree
            self._chain_metadata[chain_id]["root_hash"] = root

            logger.info(
                "构建 Merkle 树: chain=%s, leaves=%d, root=%s...",
                chain_id,
                tree.leaf_count,
                root[:16],
            )
            return root

    def get_merkle_proof(
        self,
        chain_id: str,
        node_index: int,
    ) -> list[dict[str, str]]:
        """获取指定节点的 Merkle 包含证明.

        Args:
            chain_id: 链 ID
            node_index: 节点序号

        Returns:
            Merkle 证明路径
        """
        with self._lock:
            tree = self._merkle_trees.get(chain_id)
            if tree is None or tree.leaf_count == 0:
                # 尝试构建
                self.build_merkle_tree(chain_id)
                tree = self._merkle_trees.get(chain_id)

            if tree is None:
                return []

            return tree.get_proof(node_index)

    def verify_merkle_proof(
        self,
        chain_id: str,
        node_index: int,
        root_hash: str | None = None,
    ) -> bool:
        """验证节点的 Merkle 包含证明.

        Args:
            chain_id: 链 ID
            node_index: 节点序号
            root_hash: 预期的根哈希 (None=使用当前根)

        Returns:
            证明是否有效
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            if node_index < 0 or node_index >= len(chain):
                return False

            tree = self._merkle_trees.get(chain_id)
            if tree is None or tree.leaf_count == 0:
                self.build_merkle_tree(chain_id)
                tree = self._merkle_trees.get(chain_id)

            if tree is None:
                return False

            node_hash = chain[node_index].node_hash
            proof = tree.get_proof(node_index)
            expected_root = root_hash or tree.root

            return tree.verify_proof(node_hash, proof, expected_root)

    # ==========================================================
    # 链压缩
    # ==========================================================

    def compress(self, chain_id: str) -> dict[str, Any]:
        """压缩溯源链为 Merkle 根摘要.

        将整条链压缩为:
        - Merkle 根哈希 (代表全链完整性)
        - 链元数据 (节点数、时间范围、Agent 列表)
        - 可选: 保留头尾节点用于快速验证

        Args:
            chain_id: 链 ID

        Returns:
            压缩摘要::

                {
                    "chain_id": str,
                    "merkle_root": str,
                    "node_count": int,
                    "time_range": [start, end],
                    "agents": [str, ...],
                    "layers": [str, ...],
                    "head_node_hash": str,
                    "tail_node_hash": str,
                    "compressed_at": float,
                }
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            if not chain:
                return {
                    "chain_id": chain_id,
                    "merkle_root": "",
                    "node_count": 0,
                    "time_range": [0.0, 0.0],
                    "agents": [],
                    "layers": [],
                    "head_node_hash": "",
                    "tail_node_hash": "",
                    "compressed_at": time.time(),
                }

            # 确保 Merkle 树是最新的
            root = self.build_merkle_tree(chain_id)

            timestamps = [n.timestamp for n in chain]
            agents = list({n.agent_id for n in chain if n.agent_id})
            layers = list({n.layer for n in chain if n.layer})

            return {
                "chain_id": chain_id,
                "merkle_root": root,
                "node_count": len(chain),
                "time_range": [min(timestamps), max(timestamps)],
                "agents": agents,
                "layers": layers,
                "head_node_hash": chain[0].node_hash,
                "tail_node_hash": chain[-1].node_hash,
                "compressed_at": time.time(),
            }

    # ==========================================================
    # 快照
    # ==========================================================

    def snapshot(self, chain_id: str) -> dict[str, Any]:
        """创建链快照.

        快照包含:
        - 所有节点的序列化数据
        - Merkle 树序列化
        - 链元数据
        - 验证状态

        Args:
            chain_id: 链 ID

        Returns:
            快照字典
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")

            chain = self._chains[chain_id]
            tree = self._merkle_trees.get(chain_id)

            # 确保 Merkle 树已构建
            if tree is None or tree.leaf_count == 0:
                self.build_merkle_tree(chain_id)
                tree = self._merkle_trees.get(chain_id)

            return {
                "chain_id": chain_id,
                "nodes": [n.model_dump() for n in chain],
                "merkle_tree": tree.serialize() if tree else {},
                "metadata": dict(self._chain_metadata.get(chain_id, {})),
                "snapshot_at": time.time(),
            }

    # ==========================================================
    # 审计验证
    # ==========================================================

    def audit_verify(
        self,
        chain_id: str,
        scope: str = "chain",
    ) -> AuditVerificationResult:
        """执行审计级验证.

        综合验证:
        - 哈希链完整性 (所有 prev_hash 链接)
        - Actor 一致性 (Agent ID 非空)
        - 时间戳单调递增

        Args:
            chain_id: 链 ID
            scope: 验证范围描述

        Returns:
            AuditVerificationResult
        """
        with self._lock:
            chain_report = self.verify_chain(chain_id)
            chain = self._chains.get(chain_id, [])

            # Actor 一致性: 所有节点都有 agent_id
            actor_consistency = all(n.agent_id for n in chain) if chain else True

            return AuditVerificationResult(
                scope=scope,
                scope_id=chain_id,
                total_records=len(chain),
                passed_records=chain_report["passed_nodes"],
                failed_records=chain_report["failed_nodes"],
                hash_chain_verified=chain_report["hash_chain_verified"],
                actor_consistency_verified=actor_consistency,
                timestamp_monotonic=chain_report["timestamp_monotonic"],
                pass_rate=(
                    chain_report["passed_nodes"] / len(chain)
                    if chain
                    else 1.0
                ),
                failures=chain_report["failures"],
            )

    # ==========================================================
    # 查询
    # ==========================================================

    def get_chain(self, chain_id: str) -> list[ProvenanceChainNode]:
        """获取整条链."""
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"链不存在: {chain_id}")
            return list(self._chains[chain_id])

    def get_node(
        self,
        chain_id: str,
        node_index: int,
    ) -> ProvenanceChainNode:
        """获取指定节点."""
        with self._lock:
            chain = self._chains.get(chain_id, [])
            if node_index < 0 or node_index >= len(chain):
                raise ValueError(f"节点不存在: index={node_index}")
            return chain[node_index]

    def get_chain_length(self, chain_id: str) -> int:
        """获取链长度."""
        with self._lock:
            return len(self._chains.get(chain_id, []))

    def list_chains(self) -> list[dict[str, Any]]:
        """列出所有链及其元数据."""
        with self._lock:
            return [
                dict(meta)
                for meta in self._chain_metadata.values()
            ]

    # ==========================================================
    # 跨层传递追踪
    # ==========================================================

    def trace_cross_layer(
        self,
        chain_id: str,
        target_id: str = "",
    ) -> dict[str, Any]:
        """追踪跨层传递路径.

        分析溯源链中所有跨层传递节点,
        构建跨层传递路径图。

        Args:
            chain_id: 链 ID
            target_id: 可选, 仅追踪指定对象

        Returns:
            跨层传递追踪报告::

                {
                    "chain_id": str,
                    "total_cross_layer_nodes": int,
                    "directions": {direction: count},
                    "path": [{from_layer, to_layer, agent, timestamp}],
                    "layers_involved": [str, ...],
                }
        """
        with self._lock:
            chain = self._chains.get(chain_id, [])

            cross_layer_nodes = [
                n for n in chain
                if n.direction is not None
                and (not target_id or n.target_id == target_id)
            ]

            directions: dict[str, int] = {}
            path: list[dict[str, Any]] = []
            layers: set[str] = set()

            for node in cross_layer_nodes:
                dir_name = node.direction.value if node.direction else "unknown"
                directions[dir_name] = directions.get(dir_name, 0) + 1

                # 解析方向获取 from/to 层
                if "_to_" in dir_name:
                    parts = dir_name.split("_to_")
                    from_layer = parts[0].upper()
                    to_layer = parts[1].upper()
                else:
                    from_layer = node.layer
                    to_layer = ""

                path.append({
                    "from_layer": from_layer,
                    "to_layer": to_layer,
                    "agent": node.agent_id,
                    "timestamp": node.timestamp,
                    "annotation_id": node.annotation_id,
                })

                if node.layer:
                    layers.add(node.layer)

            return {
                "chain_id": chain_id,
                "total_cross_layer_nodes": len(cross_layer_nodes),
                "directions": directions,
                "path": path,
                "layers_involved": sorted(layers),
            }

    # ==========================================================
    # 清空 (测试用)
    # ==========================================================

    def clear(self) -> None:
        """清空所有链."""
        with self._lock:
            self._chains.clear()
            self._merkle_trees.clear()
            self._chain_metadata.clear()


__all__ = [
    "MerkleTree",
    "ProvenanceChainBuilder",
]
