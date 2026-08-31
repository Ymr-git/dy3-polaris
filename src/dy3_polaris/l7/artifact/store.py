"""L7 Artifact 管理系统 — 三级存储 (store.py).

任务拆分 T3 · 设计文档 Ch.3.5。

实现 Artifact 的三级存储策略:

    L1 内存     — 当前会话活跃 Artifact (LRU 淘汰, 页面关闭即释放)
    L2 本地     — 本地持久化 (模拟 IndexedDB, LRU 淘汰 + 容量上限)
    L3 服务端   — 完整归档 (可插拔后端, 永久保留)

读取优先级: L1 > L2 > L3 (read-through 读穿)。
写策略: 元数据 write-through (保证不丢), payload 由 CAS 内容寻址去重。

融合世界先进方案:
    - 缓存分层 (Cornell CS3410): L1/L2/L3 层级 + 读穿/写穿策略
    - LRU 淘汰 (OrderedDict O(1)): 哈希表 + 访问序
    - Content-Addressable Storage (Git/IPFS): payload 按内容哈希寻址去重
    - 领域存储抽象 (L5 ArtifactStore / kernel_persistence CheckpointStore 模式)

设计要点:
    - TieredArtifactStore 是门面, 内部组合 L1/L2/L3 三层
    - 序列化快照 (save_snapshot/load_snapshot) 支持进程重启恢复
    - CAS 层 (ContentStore) 与元数据层分离: payload 按 sha256 去重
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from ..models import Artifact

# ============================================================
# CAS 内容寻址存储
# ============================================================


class ContentStore:
    """内容寻址存储 (CAS) — payload 按 sha256 去重.

    借鉴 Git blob / IPFS blockstore: 内容即地址, 相同内容只存一份,
    完整性由哈希自证。

    Attributes:
        capacity: 最大条目数 (LRU 淘汰), None 表示不限制。
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._objects: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.RLock()

    def put(self, content: Any) -> str:
        """写入内容, 返回内容哈希.

        Args:
            content: 任意可 JSON 序列化内容。

        Returns:
            sha256 十六进制哈希。
        """
        blob = self._serialize(content)
        digest = hashlib.sha256(blob).hexdigest()
        with self._lock:
            if digest not in self._objects:
                self._objects[digest] = {"blob": blob, "refs": 0}
            self._objects[digest]["refs"] += 1
            self._objects.move_to_end(digest)
            self._evict_if_needed()
        return digest

    def get(self, digest: str) -> Any | None:
        """按哈希读取内容.

        Args:
            digest: 内容哈希。

        Returns:
            反序列化后的内容, 不存在返回 None。
        """
        with self._lock:
            entry = self._objects.get(digest)
            if entry is None:
                return None
            self._objects.move_to_end(digest)
            return self._deserialize(entry["blob"])

    def contains(self, digest: str) -> bool:
        """判断内容是否存在."""
        with self._lock:
            return digest in self._objects

    def release(self, digest: str) -> None:
        """释放一次引用 (GC 辅助).

        Args:
            digest: 内容哈希。
        """
        with self._lock:
            entry = self._objects.get(digest)
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                self._objects.pop(digest, None)

    def size(self) -> int:
        """当前对象条目数."""
        with self._lock:
            return len(self._objects)

    def clear(self) -> None:
        """清空全部对象."""
        with self._lock:
            self._objects.clear()

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    @staticmethod
    def _serialize(content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, default=str).encode("utf-8")

    @staticmethod
    def _deserialize(blob: bytes) -> Any:
        try:
            return json.loads(blob.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return blob

    def _evict_if_needed(self) -> None:
        """LRU 淘汰最久未使用对象 (超出容量时)."""
        if self._capacity is None:
            return
        while len(self._objects) > self._capacity:
            self._objects.popitem(last=False)


# ============================================================
# 存储层抽象
# ============================================================


class ArtifactStore(ABC):
    """Artifact 存储层抽象接口.

    设计文档 Ch.3.5 三级存储的统一契约。
    """

    @abstractmethod
    def save(self, artifact: Artifact) -> None:
        """保存 Artifact."""

    @abstractmethod
    def load(self, artifact_id: str) -> Artifact | None:
        """按 ID 加载 Artifact."""

    @abstractmethod
    def list(self) -> list[Artifact]:
        """列出该层保存的全部 Artifact."""

    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """删除 Artifact (返回是否存在)."""

    def contains(self, artifact_id: str) -> bool:
        """判断 Artifact 是否存在."""
        return self.load(artifact_id) is not None


# ============================================================
# L1 内存层
# ============================================================


class MemoryArtifactStore(ArtifactStore):
    """L1 内存层 — 当前会话活跃 Artifact (LRU 上限可选)."""

    def __init__(self, capacity: int | None = None) -> None:
        self._items: OrderedDict[str, Artifact] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.RLock()

    def save(self, artifact: Artifact) -> None:
        with self._lock:
            self._items[artifact.artifact_id] = artifact
            self._items.move_to_end(artifact.artifact_id)
            self._evict_if_needed()

    def load(self, artifact_id: str) -> Artifact | None:
        with self._lock:
            artifact = self._items.get(artifact_id)
            if artifact is not None:
                self._items.move_to_end(artifact_id)
            return artifact

    def list(self) -> list[Artifact]:
        with self._lock:
            return list(self._items.values())

    def delete(self, artifact_id: str) -> bool:
        with self._lock:
            return self._items.pop(artifact_id, None) is not None

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _evict_if_needed(self) -> None:
        """LRU 淘汰最久未使用 (超出容量时)."""
        if self._capacity is None:
            return
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)


# ============================================================
# L2 本地层 (文件持久化, 模拟 IndexedDB)
# ============================================================

#: L2 默认容量上限 (字节, 模拟 IndexedDB 200MB)
_DEFAULT_L2_CAPACITY_BYTES: int = 200 * 1024 * 1024


class JsonFileArtifactStore(ArtifactStore):
    """L2 本地层 — 文件 JSON 持久化 (模拟 IndexedDB).

    - 元数据与 payload 以 JSON 快照文件保存, 进程重启后可恢复
    - LRU 淘汰 + 字节容量上限 (模拟 IndexedDB LRU 200MB)
    - 线程安全

    Attributes:
        path: 快照文件路径。
        capacity_bytes: 容量上限 (字节)。
        max_sessions: 保留的最大会话数 (设计文档: 最近 50 会话)。
    """

    def __init__(
        self,
        path: str,
        capacity_bytes: int = _DEFAULT_L2_CAPACITY_BYTES,
        max_sessions: int = 50,
        auto_load: bool = True,
    ) -> None:
        self.path = path
        self.capacity_bytes = capacity_bytes
        self.max_sessions = max_sessions
        self._items: dict[str, Artifact] = {}
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()
        if auto_load and os.path.exists(path):
            self._load_from_disk()

    def save(self, artifact: Artifact) -> None:
        with self._lock:
            self._items[artifact.artifact_id] = artifact
            self._touch(artifact.artifact_id)
            self._enforce_session_limit()
            self._flush()

    def load(self, artifact_id: str) -> Artifact | None:
        with self._lock:
            artifact = self._items.get(artifact_id)
            if artifact is not None:
                self._touch(artifact_id)
            return artifact

    def list(self) -> list[Artifact]:
        with self._lock:
            return list(self._items.values())

    def delete(self, artifact_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(artifact_id, None) is not None
            self._access_order.pop(artifact_id, None)
            if existed:
                self._flush()
            return existed

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def disk_usage_bytes(self) -> int:
        """估算当前占用字节数."""
        with self._lock:
            return len(json.dumps(self._to_payload(), ensure_ascii=False, default=str).encode("utf-8"))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._access_order.clear()
            self._flush()

    def flush(self) -> None:
        """强制落盘."""
        with self._lock:
            self._flush()

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _touch(self, artifact_id: str) -> None:
        self._access_order.pop(artifact_id, None)
        self._access_order[artifact_id] = None

    def _enforce_session_limit(self) -> None:
        """按会话数淘汰最久未访问的会话 (设计文档: 最近 50 会话)."""
        sessions: OrderedDict[str, list[str]] = OrderedDict()
        for aid in self._access_order:
            art = self._items.get(aid)
            if art is None:
                continue
            sid = art.session_id or "_default"
            sessions.setdefault(sid, []).append(aid)
        while len(sessions) > self.max_sessions:
            oldest_sid, aids = sessions.popitem(last=False)
            for aid in aids:
                self._items.pop(aid, None)
                self._access_order.pop(aid, None)

    def _to_payload(self) -> dict[str, Any]:
        return {
            "artifacts": {aid: art.to_dict() for aid, art in self._items.items()},
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> dict[str, Artifact]:
        items: dict[str, Artifact] = {}
        for aid, data in (payload.get("artifacts") or {}).items():
            try:
                items[aid] = Artifact.model_validate(data)
            except Exception:  # noqa: BLE001 — 单条损坏不阻断整体恢复
                continue
        return items

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._to_payload(), fh, ensure_ascii=False, default=str)

    def _load_from_disk(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        self._items = self._from_payload(payload)
        for aid in self._items:
            self._touch(aid)


# ============================================================
# L3 服务端层 (可插拔后端)
# ============================================================


class ServerArtifactStore(ABC):
    """L3 服务端存储抽象 — 永久归档 (设计文档: PostgreSQL + S3).

    提供与 ArtifactStore 一致的接口, 由调用方提供具体实现
    (PostgreSQL/S3/远程 API 等)。
    """

    @abstractmethod
    def save(self, artifact: Artifact) -> None: ...

    @abstractmethod
    def load(self, artifact_id: str) -> Artifact | None: ...

    @abstractmethod
    def list(self) -> list[Artifact]: ...

    @abstractmethod
    def delete(self, artifact_id: str) -> bool: ...


class NoopServerStore(ServerArtifactStore):
    """空实现 — 未配置服务端时使用 (L3 层直通忽略)."""

    def __init__(self, mirror_to: ArtifactStore | None = None) -> None:
        self._mirror = mirror_to

    def save(self, artifact: Artifact) -> None:
        if self._mirror is not None:
            self._mirror.save(artifact)

    def load(self, artifact_id: str) -> Artifact | None:
        if self._mirror is not None:
            return self._mirror.load(artifact_id)
        return None

    def list(self) -> list[Artifact]:
        if self._mirror is not None:
            return self._mirror.list()
        return []

    def delete(self, artifact_id: str) -> bool:
        if self._mirror is not None:
            return self._mirror.delete(artifact_id)
        return False


# ============================================================
# 三级存储门面
# ============================================================


class TieredArtifactStore(ArtifactStore):
    """三级存储门面 — 组合 L1/L2/L3, 读穿优先级 L1>L2>L3.

    使用示例::

        store = TieredArtifactStore(
            l1=MemoryArtifactStore(capacity=100),
            l2=JsonFileArtifactStore("artifacts.json"),
            l3=NoopServerStore(),
        )
        store.save(artifact)
        restored = store.load("art-xxx")   # L1 → L2 → L3 读穿

    写策略: 元数据 write-through (L1+L2 同步写, L3 尽力写)。

    Attributes:
        l1: L1 内存层。
        l2: L2 本地层 (可为 None 表示未启用)。
        l3: L3 服务端层 (可为 None 表示未启用)。
        content_store: CAS 内容层 (payload 去重)。
    """

    def __init__(
        self,
        l1: ArtifactStore | None = None,
        l2: ArtifactStore | None = None,
        l3: ServerArtifactStore | None = None,
        content_store: ContentStore | None = None,
    ) -> None:
        self.l1 = l1 or MemoryArtifactStore()
        self.l2 = l2
        self.l3 = l3
        self.content_store = content_store or ContentStore()

    # ----------------------------------------------------------
    # 读写
    # ----------------------------------------------------------

    def save(self, artifact: Artifact) -> None:
        """保存 Artifact (写穿 L1 → L2 → L3).

        若启用 CAS, payload 以内容哈希引用存储并去重。
        """
        if self.content_store is not None:
            digest = self.content_store.put(artifact.payload)
            # 保留 payload 同时记录内容哈希元数据
            meta = dict(artifact.learner_context or {})
            meta["_content_hash"] = digest
            artifact.learner_context = meta

        self.l1.save(artifact)
        if self.l2 is not None:
            self.l2.save(artifact)
        if self.l3 is not None:
            try:
                self.l3.save(artifact)
            except Exception:  # noqa: BLE001 — L3 不可用不阻断本地写
                pass

    def load(self, artifact_id: str) -> Artifact | None:
        """读穿加载: L1 → L2 → L3, 命中后回填近层."""
        artifact = self.l1.load(artifact_id)
        if artifact is not None:
            return artifact
        if self.l2 is not None:
            artifact = self.l2.load(artifact_id)
            if artifact is not None:
                # 回填 L1 (read-through)
                self.l1.save(artifact)
                return artifact
        if self.l3 is not None:
            artifact = self.l3.load(artifact_id)
            if artifact is not None:
                self.l1.save(artifact)
                if self.l2 is not None:
                    self.l2.save(artifact)
                return artifact
        return None

    def list(self) -> list[Artifact]:
        """列出全部 Artifact (L1 优先, 合并 L2/L3 去重)."""
        seen: dict[str, Artifact] = {}
        for art in self.l1.list():
            seen[art.artifact_id] = art
        if self.l2 is not None:
            for art in self.l2.list():
                seen.setdefault(art.artifact_id, art)
        if self.l3 is not None:
            try:
                for art in self.l3.list():
                    seen.setdefault(art.artifact_id, art)
            except Exception:  # noqa: BLE001
                pass
        return list(seen.values())

    def delete(self, artifact_id: str) -> bool:
        """从全部层删除 Artifact."""
        existed = self.l1.delete(artifact_id)
        if self.l2 is not None:
            existed = self.l2.delete(artifact_id) or existed
        if self.l3 is not None:
            try:
                existed = self.l3.delete(artifact_id) or existed
            except Exception:  # noqa: BLE001
                pass
        return existed

    def contains(self, artifact_id: str) -> bool:
        return self.load(artifact_id) is not None

    # ----------------------------------------------------------
    # 快照 (进程重启恢复)
    # ----------------------------------------------------------

    def save_snapshot(self, path: str) -> int:
        """将当前全部 Artifact 序列化保存为快照文件.

        Args:
            path: 快照文件路径。

        Returns:
            保存的 Artifact 数量。
        """
        artifacts = self.list()
        payload = {"artifacts": {a.artifact_id: a.to_dict() for a in artifacts}}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str)
        return len(artifacts)

    def load_snapshot(self, path: str) -> int:
        """从快照文件恢复 Artifact (L1 + L2 回填).

        Args:
            path: 快照文件路径。

        Returns:
            恢复的 Artifact 数量。

        Raises:
            FileNotFoundError: 快照文件不存在。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"快照文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        count = 0
        for aid, data in (payload.get("artifacts") or {}).items():
            try:
                artifact = Artifact.model_validate(data)
            except Exception:  # noqa: BLE001 — 单条损坏不阻断整体恢复
                continue
            self.l1.save(artifact)
            if self.l2 is not None:
                self.l2.save(artifact)
            count += 1
        return count
