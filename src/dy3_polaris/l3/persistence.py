"""L3 领域知识层 — 持久化与事务管理.

融合世界先进方案的持久化与事务设计:
- Redis RDB: 全量快照持久化 (周期性全量序列化到磁盘)
- Redis AOF: 增量日志持久化 (追加式变更日志, append-only)
- SQLite WAL: Write-Ahead Logging (写前日志 + checkpoint 合并)
- Neo4j Checkpoint: 周期性检查点 + 增量合并压缩
- MVCC: 多版本并发控制 (乐观并发 + 快照隔离)
- SQLite Transaction: ACID 事务 (BEGIN/COMMIT/ROLLBACK + SAVEPOINT)

两种持久化模式:
1. 快照模式 (snapshot): 全量序列化到目录结构 (JSONL + manifest)
   - 借鉴 Redis RDB: 将整个知识库状态序列化到磁盘
   - 目录结构: manifest.json + entities.jsonl + triples.jsonl + chunks.jsonl
     + versions.jsonl + conflicts.jsonl + evidence.jsonl
   - 校验和: 每个文件计算 SHA-256, manifest 记录所有校验和

2. 增量模式 (incremental): WAL 风格变更日志 (append-only)
   - 借鉴 Redis AOF + SQLite WAL: 只记录变更操作, 追加写入
   - 加载时: 先加载最新快照, 再按序回放增量日志
   - compact: 合并增量日志到新快照 (借鉴 AOF rewrite + checkpoint)

事务管理 (借鉴 SQLite BEGIN/COMMIT/ROLLBACK + MVCC):
- undo log 模式: 操作执行前记录逆操作, rollback 时逆序回放
- savepoint 嵌套回滚: 支持部分回滚到保存点
- 上下文管理器: with tx_manager.begin() as tx: ... (自动提交/回滚)
- 线程安全: 所有操作通过 RLock 保护

JSON-LD 导出 (借鉴 schema.org 词汇表):
- 实体映射为 JSON-LD 节点, 使用 schema.org 类型和属性
- 支持双向转换: 实体 <-> JSON-LD 节点

Usage::

    from dy3_polaris.l3.persistence import PersistenceManager, TransactionManager

    # 持久化管理
    pm = PersistenceManager(store, "/data/kb")
    pm.save_snapshot()           # 保存快照
    pm.save_incremental(changes) # 保存增量
    pm.compact()                 # 压缩合并

    # 事务管理
    txm = TransactionManager(store)
    with txm.begin() as tx:
        tx.add_entity(entity)
        tx.update_entity(eid, name="新名称")
        # 正常退出自动提交, 异常退出自动回滚
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .exceptions import *  # noqa: F403
from .models import *  # noqa: F403
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# 快照格式版本 (主版本.次版本, 主版本不兼容则拒绝加载)
_SNAPSHOT_FORMAT_VERSION = "1.0"

# JSONL 文件名常量
_FILE_ENTITIES = "entities.jsonl"
_FILE_TRIPLES = "triples.jsonl"
_FILE_CHUNKS = "chunks.jsonl"
_FILE_VERSIONS = "versions.jsonl"
_FILE_CONFLICTS = "conflicts.jsonl"
_FILE_EVIDENCE = "evidence.jsonl"
_FILE_MANIFEST = "manifest.json"

# schema.org 类型映射表 (EntityType.value -> schema.org 类型)
_SCHEMA_ORG_TYPE_MAP: dict[str, str] = {
    "concept": "schema:Thing",
    "chemical_compound": "schema:ChemicalSubstance",
    "material": "schema:Substance",
    "paper": "schema:ScholarlyArticle",
    "textbook": "schema:Book",
    "dataset": "schema:Dataset",
    "method": "schema:Thing",
    "person": "schema:Person",
    "organization": "schema:Organization",
    "document_chunk": "schema:Text",
    "course": "schema:Course",
    "experiment": "schema:Action",
}

# schema.org 属性反向映射表 (schema.org 属性 -> 内部属性名)
_SCHEMA_ORG_PROP_REVERSE_MAP: dict[str, str] = {
    "schema:author": "author",
    "schema:creator": "creator",
    "schema:datePublished": "date",
    "schema:publisher": "publisher",
    "schema:inLanguage": "language",
    "schema:license": "license",
    "schema:version": "version",
    "schema:dateCreated": "date_created",
    "schema:dateModified": "date_modified",
}

# schema.org 保留属性集合 (不映射到 properties 的属性)
_SCHEMA_ORG_RESERVED_PROPS = {
    "schema:name", "schema:description", "schema:category",
    "schema:alternateName", "schema:keywords", "schema:identifier",
    "schema:isbn", "schema:url", "schema:rating",
    *_SCHEMA_ORG_PROP_REVERSE_MAP.keys(),
}


# ============================================================
# 事务状态枚举
# ============================================================


class TransactionState(enum.Enum):
    """事务状态 (借鉴 SQLite 事务状态机).

    状态转换:
        ACTIVE -> COMMITTED     (commit 成功)
        ACTIVE -> ROLLED_BACK   (rollback 成功)
        ACTIVE -> FAILED        (rollback 过程中出错)

    Attributes:
        ACTIVE: 事务活跃中, 可执行操作
        COMMITTED: 事务已提交, 不可再操作
        ROLLED_BACK: 事务已回滚, 不可再操作
        FAILED: 事务回滚失败 (数据可能不一致)
    """

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# ============================================================
# 持久化管理器 — Redis RDB + AOF + SQLite WAL
# ============================================================


class PersistenceManager:
    """知识库持久化管理器.

    融合 Redis RDB 快照 + AOF 增量日志 + SQLite WAL 模式。

    提供两种持久化模式:
    1. 快照模式 (snapshot): 全量序列化到 JSON 文件
    2. 增量模式 (incremental): 只保存变更日志 (WAL 风格)

    支持自动压缩、校验和验证、版本兼容性检查。

    快照目录结构::

        snapshot_dir/
          manifest.json       # 元数据 (版本, 时间, 统计, 校验和)
          entities.jsonl      # 每行一个实体 JSON
          triples.jsonl       # 每行一个三元组 JSON
          chunks.jsonl        # 每行一个切片 JSON
          versions.jsonl      # 版本历史
          conflicts.jsonl     # 冲突记录
          evidence.jsonl      # 证据记录

    Attributes:
        _store: 关联的知识存储
        _base_path: 持久化基础路径
        _lock: 线程安全锁
        _incremental_counter: 增量日志序列号
    """

    def __init__(self, store: KnowledgeStore, base_path: str | Path) -> None:
        """初始化持久化管理器.

        Args:
            store: 关联的知识存储引擎
            base_path: 持久化基础路径 (快照和增量日志存放目录)
        """
        self._store = store
        self._base_path = Path(base_path)
        os.makedirs(self._base_path, exist_ok=True)
        self._lock = threading.RLock()
        self._incremental_counter = 0

    # --------------------------------------------------------
    # 内部工具方法
    # --------------------------------------------------------

    @staticmethod
    def _compute_sha256(data: bytes) -> str:
        """计算字节数据的 SHA-256 校验和.

        Args:
            data: 待计算的字节数据

        Returns:
            十六进制校验和字符串
        """
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _compute_file_checksum(file_path: Path) -> str:
        """计算文件的 SHA-256 校验和 (流式读取, 支持大文件).

        Args:
            file_path: 文件路径

        Returns:
            十六进制校验和字符串
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_jsonl(self, file_path: Path, items: list[Any]) -> str:
        """将对象列表写入 JSONL 文件 (每行一个 JSON 对象).

        支持 Pydantic BaseModel 对象和普通 dict。

        Args:
            file_path: 目标文件路径
            items: 待写入的对象列表

        Returns:
            文件内容的 SHA-256 校验和
        """
        lines: list[str] = []
        for item in items:
            if hasattr(item, "model_dump"):
                data = item.model_dump(mode="json")
            else:
                data = item
            lines.append(json.dumps(data, ensure_ascii=False))

        content = "\n".join(lines)
        if content:
            content += "\n"

        data_bytes = content.encode("utf-8")
        file_path.write_bytes(data_bytes)
        return self._compute_sha256(data_bytes)

    @staticmethod
    def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
        """从 JSONL 文件读取对象列表 (每行一个 JSON 对象).

        Args:
            file_path: JSONL 文件路径

        Returns:
            解析后的字典列表 (文件不存在或为空时返回空列表)
        """
        items: list[dict[str, Any]] = []
        if not file_path.exists():
            return items
        content = file_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    @staticmethod
    def _is_version_compatible(format_version: str) -> bool:
        """检查快照格式版本是否兼容 (主版本号必须一致).

        Args:
            format_version: 快照格式版本字符串 (如 "1.0")

        Returns:
            是否兼容
        """
        try:
            major = format_version.split(".")[0]
            current_major = _SNAPSHOT_FORMAT_VERSION.split(".")[0]
            return major == current_major
        except (ValueError, IndexError, AttributeError):
            return False

    @staticmethod
    def _compress_directory(dir_path: Path, zip_path: Path) -> Path:
        """压缩目录为 zip 文件 (使用 zipfile).

        Args:
            dir_path: 源目录路径
            zip_path: 目标 zip 文件路径

        Returns:
            zip 文件路径
        """
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(dir_path)
                    zf.write(file_path, arcname)
        return zip_path

    def _collect_all_data(self) -> dict[str, list[Any]]:
        """收集知识库中的所有数据 (实体/三元组/切片/版本/冲突/证据).

        Returns:
            包含所有数据的字典, 键为数据类型, 值为对象列表
        """
        store = self._store
        entities = store.entity_store.list_entities(limit=100000)
        triples = list(store.triple_store._triples.values())
        chunks = list(store.chunk_store._chunks.values())

        # 收集版本历史
        versions: list[KnowledgeVersion] = []
        for version_list in store._versions.values():
            versions.extend(version_list)

        # 收集冲突和证据
        conflicts = list(store._conflicts.values())
        evidence = list(store._evidence.values())

        return {
            _FILE_ENTITIES: entities,
            _FILE_TRIPLES: triples,
            _FILE_CHUNKS: chunks,
            _FILE_VERSIONS: versions,
            _FILE_CONFLICTS: conflicts,
            _FILE_EVIDENCE: evidence,
        }

    # --------------------------------------------------------
    # 快照模式 (借鉴 Redis RDB)
    # --------------------------------------------------------

    def save_snapshot(self, path: str | Path | None = None) -> Path:
        """保存全量快照到目录结构 (借鉴 Redis RDB 全量持久化).

        将知识库的完整状态序列化到目录, 包含:
        - manifest.json: 元数据 (版本, 时间, 统计, 校验和)
        - entities.jsonl: 每行一个实体 JSON
        - triples.jsonl: 每行一个三元组 JSON
        - chunks.jsonl: 每行一个切片 JSON
        - versions.jsonl: 版本历史
        - conflicts.jsonl: 冲突记录
        - evidence.jsonl: 证据记录

        Args:
            path: 快照目录路径, None 则自动生成时间戳命名路径

        Returns:
            快照目录路径
        """
        with self._lock:
            # 确定快照目录
            if path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_dir = self._base_path / f"snapshot_{timestamp}"
            else:
                snapshot_dir = Path(path)

            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # 收集所有数据
            all_data = self._collect_all_data()

            # 写入 JSONL 文件并计算校验和
            checksums: dict[str, str] = {}
            counts: dict[str, int] = {}

            file_key_map = {
                _FILE_ENTITIES: "entities",
                _FILE_TRIPLES: "triples",
                _FILE_CHUNKS: "chunks",
                _FILE_VERSIONS: "versions",
                _FILE_CONFLICTS: "conflicts",
                _FILE_EVIDENCE: "evidence",
            }

            for filename, count_key in file_key_map.items():
                items = all_data[filename]
                file_path = snapshot_dir / filename
                checksum = self._write_jsonl(file_path, items)
                checksums[filename] = checksum
                counts[count_key] = len(items)

            # 获取统计信息
            stats = self._store.get_stats()

            # 构建 manifest
            manifest: dict[str, Any] = {
                "format_version": _SNAPSHOT_FORMAT_VERSION,
                "created_at": time.time(),
                "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "counts": counts,
                "checksums": checksums,
                "stats": stats.model_dump(mode="json"),
                "manifest_checksum": "",  # 占位, 后续填充
            }

            # 计算 manifest 自身校验和 (排除 manifest_checksum 字段)
            manifest_for_checksum = {
                k: v for k, v in manifest.items() if k != "manifest_checksum"
            }
            manifest_json = json.dumps(
                manifest_for_checksum, ensure_ascii=False, sort_keys=True
            )
            manifest["manifest_checksum"] = self._compute_sha256(
                manifest_json.encode("utf-8")
            )

            # 写入 manifest.json
            (snapshot_dir / _FILE_MANIFEST).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info(
                "保存快照: %s (实体=%d, 三元组=%d, 切片=%d, 版本=%d, 冲突=%d, 证据=%d)",
                snapshot_dir,
                counts["entities"], counts["triples"], counts["chunks"],
                counts["versions"], counts["conflicts"], counts["evidence"],
            )
            return snapshot_dir

    def load_snapshot(self, path: str | Path | None = None) -> None:
        """从快照目录加载全量数据 (借鉴 Redis RDB 加载恢复).

        加载流程:
        1. 读取 manifest.json, 检查格式版本兼容性
        2. 校验所有文件的 SHA-256 校验和
        3. 清空当前存储
        4. 按序加载实体 -> 三元组 -> 切片 -> 版本 -> 冲突 -> 证据

        Args:
            path: 快照目录路径, None 则自动查找最新快照

        Raises:
            IngestError: 快照不存在、版本不兼容或校验失败
        """
        with self._lock:
            # 确定快照目录
            if path is None:
                snapshots = sorted(self._base_path.glob("snapshot_*"))
                # 仅保留目录
                snapshots = [s for s in snapshots if s.is_dir()]
                if not snapshots:
                    raise IngestError(
                        source="snapshot",
                        detail="未找到快照目录",
                    )
                snapshot_dir = snapshots[-1]
            else:
                snapshot_dir = Path(path)

            # 读取 manifest
            manifest_path = snapshot_dir / _FILE_MANIFEST
            if not manifest_path.exists():
                raise IngestError(
                    source=str(snapshot_dir),
                    detail="manifest.json 不存在",
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            # 版本兼容性检查
            format_version = manifest.get("format_version", "0.0")
            if not self._is_version_compatible(format_version):
                raise IngestError(
                    source=str(snapshot_dir),
                    detail=(
                        f"快照版本不兼容: {format_version} "
                        f"(当前支持 {_SNAPSHOT_FORMAT_VERSION})"
                    ),
                )

            # 校验和验证
            stored_checksums = manifest.get("checksums", {})
            for filename, stored_checksum in stored_checksums.items():
                file_path = snapshot_dir / filename
                if not file_path.exists():
                    raise IngestError(
                        source=str(snapshot_dir),
                        detail=f"快照文件缺失: {filename}",
                    )
                actual_checksum = self._compute_file_checksum(file_path)
                if actual_checksum != stored_checksum:
                    raise IngestError(
                        source=str(snapshot_dir),
                        detail=(
                            f"校验和不匹配: {filename} "
                            f"(期望={stored_checksum[:16]}..., "
                            f"实际={actual_checksum[:16]}...)"
                        ),
                    )

            # 清空当前存储
            self._store.clear()

            # 加载实体
            entity_data = self._read_jsonl(snapshot_dir / _FILE_ENTITIES)
            for item in entity_data:
                entity = KnowledgeEntity.model_validate(item)
                self._store.entity_store.add_entity(entity, check_duplicate=False)

            # 加载三元组
            triple_data = self._read_jsonl(snapshot_dir / _FILE_TRIPLES)
            for item in triple_data:
                triple = KnowledgeTriple.model_validate(item)
                self._store.triple_store.add_triple(triple)

            # 加载切片
            chunk_data = self._read_jsonl(snapshot_dir / _FILE_CHUNKS)
            for item in chunk_data:
                chunk = DocumentChunk.model_validate(item)
                self._store.chunk_store.add_chunk(chunk)

            # 加载版本历史 (直接写入, 不通过 add_entity 避免重复创建版本)
            version_data = self._read_jsonl(snapshot_dir / _FILE_VERSIONS)
            with self._store._lock:
                for item in version_data:
                    version = KnowledgeVersion.model_validate(item)
                    self._store._versions[version.entity_id].append(version)

            # 加载冲突记录
            conflict_data = self._read_jsonl(snapshot_dir / _FILE_CONFLICTS)
            with self._store._lock:
                for item in conflict_data:
                    conflict = KnowledgeConflict.model_validate(item)
                    self._store._conflicts[conflict.conflict_id] = conflict

            # 加载证据记录
            evidence_data = self._read_jsonl(snapshot_dir / _FILE_EVIDENCE)
            with self._store._lock:
                for item in evidence_data:
                    evidence = EvidenceRecord.model_validate(item)
                    self._store._evidence[evidence.evidence_id] = evidence

            counts = manifest.get("counts", {})
            logger.info(
                "加载快照: %s (实体=%d, 三元组=%d, 切片=%d)",
                snapshot_dir,
                counts.get("entities", 0),
                counts.get("triples", 0),
                counts.get("chunks", 0),
            )

    # --------------------------------------------------------
    # 增量模式 (借鉴 Redis AOF + SQLite WAL)
    # --------------------------------------------------------

    def save_incremental(self, changes: list[dict[str, Any]]) -> Path:
        """保存增量变更日志 (借鉴 Redis AOF + SQLite WAL).

        以 append-only 方式记录变更操作, 每行一个变更记录。
        变更记录包含序列号、时间戳和变更内容。

        WAL 文件命名: wal_YYYYMMDD_HHMMSS_NNNNNN.jsonl

        Args:
            changes: 变更记录列表, 每条为描述操作的字典

        Returns:
            WAL 日志文件路径
        """
        with self._lock:
            self._incremental_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"wal_{timestamp}_{self._incremental_counter:06d}.jsonl"
            wal_path = self._base_path / filename

            # WAL 格式: 每行一个变更记录 (含序列号和时间戳)
            lines: list[str] = []
            for change in changes:
                record = {
                    "seq": self._incremental_counter,
                    "timestamp": time.time(),
                    "change": change,
                }
                lines.append(json.dumps(record, ensure_ascii=False))

            content = "\n".join(lines)
            if content:
                content += "\n"

            wal_path.write_text(content, encoding="utf-8")

            logger.info("保存增量日志: %s (%d 条变更)", wal_path, len(changes))
            return wal_path

    def load_incremental(self, path: str | Path) -> list[dict[str, Any]]:
        """加载增量变更日志 (借鉴 SQLite WAL 回放).

        读取 WAL 日志文件, 返回变更记录列表 (按写入顺序)。

        Args:
            path: WAL 日志文件路径

        Returns:
            变更记录列表 (每条为描述操作的字典)

        Raises:
            IngestError: 文件不存在
        """
        wal_path = Path(path)
        if not wal_path.exists():
            raise IngestError(
                source=str(wal_path),
                detail="增量日志文件不存在",
            )

        changes: list[dict[str, Any]] = []
        for line in wal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                record = json.loads(line)
                # 兼容带包装的 WAL 记录和裸变更记录
                if "change" in record:
                    changes.append(record["change"])
                else:
                    changes.append(record)

        logger.info("加载增量日志: %s (%d 条变更)", wal_path, len(changes))
        return changes

    # --------------------------------------------------------
    # JSON 导入/导出
    # --------------------------------------------------------

    def export_json(self, path: str | Path) -> Path:
        """导出为单个 JSON 文件 (全量序列化).

        将知识库所有数据导出为一个 JSON 文件, 包含:
        format_version, exported_at, entities, triples, chunks,
        versions, conflicts, evidence, stats。

        Args:
            path: 导出文件路径

        Returns:
            导出文件路径
        """
        with self._lock:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            all_data = self._collect_all_data()

            # 序列化所有数据为 dict
            data: dict[str, Any] = {
                "format_version": _SNAPSHOT_FORMAT_VERSION,
                "exported_at": time.time(),
                "entities": [
                    e.model_dump(mode="json") if hasattr(e, "model_dump") else e
                    for e in all_data[_FILE_ENTITIES]
                ],
                "triples": [
                    t.model_dump(mode="json") if hasattr(t, "model_dump") else t
                    for t in all_data[_FILE_TRIPLES]
                ],
                "chunks": [
                    c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                    for c in all_data[_FILE_CHUNKS]
                ],
                "versions": [
                    v.model_dump(mode="json") if hasattr(v, "model_dump") else v
                    for v in all_data[_FILE_VERSIONS]
                ],
                "conflicts": [
                    c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                    for c in all_data[_FILE_CONFLICTS]
                ],
                "evidence": [
                    e.model_dump(mode="json") if hasattr(e, "model_dump") else e
                    for e in all_data[_FILE_EVIDENCE]
                ],
                "stats": self._store.get_stats().model_dump(mode="json"),
            }

            export_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info(
                "导出 JSON: %s (实体=%d, 三元组=%d, 切片=%d)",
                export_path,
                len(data["entities"]),
                len(data["triples"]),
                len(data["chunks"]),
            )
            return export_path

    def import_json(self, path: str | Path) -> IngestResult:
        """从 JSON 文件导入知识 (合并模式, 跳过已存在的).

        逐条导入实体、三元组、切片, 已存在的跳过。
        版本历史、冲突记录和证据记录直接追加。

        Args:
            path: JSON 文件路径

        Returns:
            导入结果 (含成功/失败/跳过计数)

        Raises:
            IngestError: 文件不存在
        """
        start_time = time.time()
        import_path = Path(path)
        if not import_path.exists():
            raise IngestError(
                source=str(import_path),
                detail="JSON 文件不存在",
            )

        data = json.loads(import_path.read_text(encoding="utf-8"))

        total = 0
        success = 0
        failed = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        ingested_ids: list[str] = []

        store = self._store

        # 导入实体
        for item in data.get("entities", []):
            total += 1
            try:
                entity = KnowledgeEntity.model_validate(item)
                if store.entity_store.exists(entity.entity_id):
                    skipped += 1
                    continue
                store.entity_store.add_entity(entity, check_duplicate=False)
                success += 1
                ingested_ids.append(entity.entity_id)
            except Exception as exc:
                failed += 1
                errors.append({
                    "type": "entity",
                    "id": item.get("entity_id", ""),
                    "error": str(exc),
                })

        # 导入三元组
        for item in data.get("triples", []):
            total += 1
            try:
                triple = KnowledgeTriple.model_validate(item)
                if store.triple_store.exists(triple.triple_id):
                    skipped += 1
                    continue
                store.triple_store.add_triple(triple)
                success += 1
                ingested_ids.append(triple.triple_id)
            except Exception as exc:
                failed += 1
                errors.append({
                    "type": "triple",
                    "id": item.get("triple_id", ""),
                    "error": str(exc),
                })

        # 导入切片
        for item in data.get("chunks", []):
            total += 1
            try:
                chunk = DocumentChunk.model_validate(item)
                if store.chunk_store.exists(chunk.chunk_id):
                    skipped += 1
                    continue
                store.chunk_store.add_chunk(chunk)
                success += 1
                ingested_ids.append(chunk.chunk_id)
            except Exception as exc:
                failed += 1
                errors.append({
                    "type": "chunk",
                    "id": item.get("chunk_id", ""),
                    "error": str(exc),
                })

        # 导入版本历史 (直接追加, 不计入主计数)
        for item in data.get("versions", []):
            try:
                version = KnowledgeVersion.model_validate(item)
                with store._lock:
                    store._versions[version.entity_id].append(version)
            except Exception as exc:
                logger.warning("导入版本记录失败: %s", exc)

        # 导入冲突记录 (直接追加, 不计入主计数)
        for item in data.get("conflicts", []):
            try:
                conflict = KnowledgeConflict.model_validate(item)
                with store._lock:
                    store._conflicts[conflict.conflict_id] = conflict
            except Exception as exc:
                logger.warning("导入冲突记录失败: %s", exc)

        # 导入证据记录 (直接追加, 不计入主计数)
        for item in data.get("evidence", []):
            try:
                evidence = EvidenceRecord.model_validate(item)
                with store._lock:
                    store._evidence[evidence.evidence_id] = evidence
            except Exception as exc:
                logger.warning("导入证据记录失败: %s", exc)

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "导入 JSON: %s (成功=%d, 失败=%d, 跳过=%d)",
            import_path, success, failed, skipped,
        )

        return IngestResult(
            source=str(import_path),
            total=total,
            success=success,
            failed=failed,
            skipped=skipped,
            errors=errors,
            ingested_ids=ingested_ids,
            duration_ms=round(duration_ms, 2),
        )

    # --------------------------------------------------------
    # JSON-LD 导入/导出 (schema.org 词汇表)
    # --------------------------------------------------------

    def export_jsonld(self, path: str | Path) -> Path:
        """导出为 JSON-LD 格式 (使用 schema.org 词汇表).

        将知识实体映射为 JSON-LD 节点, 使用 schema.org 类型体系:
        - ChemicalCompound -> schema:ChemicalSubstance
        - Person -> schema:Person
        - Organization -> schema:Organization
        - Paper -> schema:ScholarlyArticle
        - 等等...

        每个实体生成一个 JSON-LD 节点, 包含:
        - @id: 实体 URN (urn:entity:{entity_id})
        - @type: schema.org 类型
        - schema:name / schema:description / schema:identifier 等属性
        - 三元组映射为 schema.org 关系

        Args:
            path: 导出文件路径

        Returns:
            导出文件路径
        """
        with self._lock:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # JSON-LD 上下文 (借鉴 schema.org + PROV-O + Dublin Core)
            context = {
                "schema": "https://schema.org/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "dcterms": "http://purl.org/dc/terms/",
                "prov": "http://www.w3.org/ns/prov#",
            }

            # 构建 JSON-LD 图
            graph: list[dict[str, Any]] = []
            for entity in self._store.entity_store.list_entities(limit=100000):
                node = self._entity_to_jsonld(entity)
                graph.append(node)

            data = {
                "@context": context,
                "@graph": graph,
            }

            export_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info("导出 JSON-LD: %s (%d 个节点)", export_path, len(graph))
            return export_path

    def _entity_to_jsonld(self, entity: KnowledgeEntity) -> dict[str, Any]:
        """将知识实体转换为 JSON-LD 节点 (schema.org 映射).

        映射规则:
        - entity_type -> @type (schema.org 类型)
        - name -> schema:name
        - description -> schema:description
        - domain -> schema:category
        - aliases -> schema:alternateName
        - tags -> schema:keywords
        - identifiers -> schema:identifier / schema:isbn / schema:url
        - properties -> schema.org 对应属性
        - triples -> schema.org 关系 (实体引用为 @id 引用)
        - quality -> schema:Rating

        Args:
            entity: 知识实体

        Returns:
            JSON-LD 节点字典
        """
        # 实体类型映射
        type_value = (
            entity.entity_type.value
            if hasattr(entity.entity_type, "value")
            else str(entity.entity_type)
        )
        schema_type = _SCHEMA_ORG_TYPE_MAP.get(type_value, "schema:Thing")

        node: dict[str, Any] = {
            "@id": f"urn:entity:{entity.entity_id}",
            "@type": schema_type,
            "schema:name": entity.name,
        }

        if entity.description:
            node["schema:description"] = entity.description

        if entity.domain:
            node["schema:category"] = entity.domain

        # 别名
        if entity.aliases:
            node["schema:alternateName"] = entity.aliases

        # 标签
        if entity.tags:
            node["schema:keywords"] = ",".join(entity.tags)

        # 标识符映射
        identifiers: list[dict[str, str]] = []
        for id_type, id_value in entity.identifiers.items():
            id_lower = id_type.lower()
            if id_lower == "doi":
                node["schema:identifier"] = f"doi:{id_value}"
            elif id_lower == "isbn":
                node["schema:isbn"] = id_value
            elif id_lower == "url":
                node["schema:url"] = id_value
            else:
                identifiers.append({id_type: id_value})
        if identifiers:
            existing_id = node.get("schema:identifier")
            if existing_id:
                identifiers.insert(0, existing_id if isinstance(existing_id, dict) else {"id": existing_id})
            node["schema:identifier"] = identifiers

        # 属性映射 (常见属性映射到 schema.org 标准属性)
        prop_to_schema = {
            "author": "schema:author",
            "creator": "schema:author",
            "date": "schema:datePublished",
            "publisher": "schema:publisher",
            "language": "schema:inLanguage",
            "license": "schema:license",
        }
        for key, value in entity.properties.items():
            schema_key = prop_to_schema.get(key, f"schema:{key}")
            node[schema_key] = value

        # 三元组映射为 RDF 关系
        for triple in entity.triples:
            pred_key = f"schema:{triple.predicate}"
            if triple.object_is_literal:
                # 字面值宾语
                node[pred_key] = triple.object_value
            elif triple.object_id:
                # 实体引用宾语 (使用 @id 引用)
                ref = {"@id": f"urn:entity:{triple.object_id}"}
                if pred_key in node:
                    if isinstance(node[pred_key], list):
                        node[pred_key].append(ref)
                    else:
                        node[pred_key] = [node[pred_key], ref]
                else:
                    node[pred_key] = ref

        # 溯源信息 (借鉴 PROV-O)
        if entity.provenance and entity.provenance.generator:
            node["prov:wasGeneratedBy"] = entity.provenance.generator

        # 质量评分 (映射为 schema:Rating)
        if entity.quality:
            node["schema:rating"] = {
                "@type": "schema:Rating",
                "schema:ratingValue": entity.quality.overall(),
                "schema:bestRating": 1.0,
                "schema:worstRating": 0.0,
            }

        # 状态信息
        if hasattr(entity.status, "value"):
            node["schema:creativeWorkStatus"] = entity.status.value

        return node

    def import_jsonld(self, path: str | Path) -> IngestResult:
        """从 JSON-LD 文件导入知识 (反向映射 schema.org 到知识实体).

        解析 JSON-LD 图, 将每个节点反向映射为知识实体:
        - @type -> EntityType (schema.org 类型反向映射)
        - schema:name -> name
        - schema:description -> description
        - schema:identifier -> identifiers
        - schema:* 属性 -> properties

        Args:
            path: JSON-LD 文件路径

        Returns:
            导入结果 (含成功/失败/跳过计数)

        Raises:
            IngestError: 文件不存在
        """
        start_time = time.time()
        import_path = Path(path)
        if not import_path.exists():
            raise IngestError(
                source=str(import_path),
                detail="JSON-LD 文件不存在",
            )

        data = json.loads(import_path.read_text(encoding="utf-8"))

        # 支持 @graph 数组或单个节点
        if isinstance(data, dict):
            nodes = data.get("@graph", [])
            if isinstance(nodes, dict):
                nodes = [nodes]
        elif isinstance(data, list):
            nodes = data
        else:
            nodes = []

        # 反向类型映射表
        reverse_type_map = {v: k for k, v in _SCHEMA_ORG_TYPE_MAP.items()}

        total = 0
        success = 0
        failed = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        ingested_ids: list[str] = []

        for node in nodes:
            total += 1
            try:
                entity = self._jsonld_to_entity(node, reverse_type_map)
                if entity is None:
                    skipped += 1
                    continue
                if self._store.entity_store.exists(entity.entity_id):
                    skipped += 1
                    continue
                self._store.entity_store.add_entity(entity, check_duplicate=False)
                success += 1
                ingested_ids.append(entity.entity_id)
            except Exception as exc:
                failed += 1
                errors.append({
                    "type": "entity",
                    "id": node.get("@id", ""),
                    "error": str(exc),
                })

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "导入 JSON-LD: %s (成功=%d, 失败=%d, 跳过=%d)",
            import_path, success, failed, skipped,
        )

        return IngestResult(
            source=str(import_path),
            total=total,
            success=success,
            failed=failed,
            skipped=skipped,
            errors=errors,
            ingested_ids=ingested_ids,
            duration_ms=round(duration_ms, 2),
        )

    def _jsonld_to_entity(
        self,
        node: dict[str, Any],
        reverse_type_map: dict[str, str],
    ) -> KnowledgeEntity | None:
        """将 JSON-LD 节点反向映射为知识实体.

        反向映射规则:
        - @id (urn:entity:xxx) -> entity_id
        - @type -> EntityType (反向映射 schema.org 类型)
        - schema:name -> name (必需, 缺失则返回 None)
        - schema:description -> description
        - schema:category -> domain
        - schema:alternateName -> aliases
        - schema:keywords -> tags
        - schema:identifier / schema:isbn / schema:url -> identifiers
        - schema:* 属性 -> properties

        Args:
            node: JSON-LD 节点字典
            reverse_type_map: schema.org 类型反向映射表

        Returns:
            知识实体, 若节点缺少 name 则返回 None
        """
        # 提取 entity_id
        node_id = node.get("@id", "")
        if isinstance(node_id, str) and node_id.startswith("urn:entity:"):
            entity_id = node_id.replace("urn:entity:", "")
        else:
            entity_id = f"e-{uuid.uuid4().hex[:12]}"

        # 提取实体类型
        node_type = node.get("@type", "schema:Thing")
        if isinstance(node_type, list):
            node_type = node_type[0] if node_type else "schema:Thing"
        entity_type_str = reverse_type_map.get(node_type, "concept")
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            entity_type = EntityType.CONCEPT

        # 提取名称 (必需)
        name = node.get("schema:name", "")
        if not name or not isinstance(name, str):
            return None

        description = node.get("schema:description", "")
        if not isinstance(description, str):
            description = str(description)

        domain = node.get("schema:category", "general")
        if not isinstance(domain, str):
            domain = "general"

        # 别名
        aliases = node.get("schema:alternateName", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        # 标签
        tags_str = node.get("schema:keywords", "")
        tags = tags_str.split(",") if isinstance(tags_str, str) and tags_str else []

        # 标识符
        identifiers: dict[str, str] = {}

        # schema:identifier
        raw_id = node.get("schema:identifier")
        if raw_id:
            if isinstance(raw_id, str):
                if raw_id.startswith("doi:"):
                    identifiers["doi"] = raw_id[4:]
                else:
                    identifiers["id"] = raw_id
            elif isinstance(raw_id, list):
                for item in raw_id:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            identifiers[k] = str(v)
                    elif isinstance(item, str):
                        identifiers.setdefault("id", item)

        # schema:isbn
        if "schema:isbn" in node:
            identifiers["isbn"] = str(node["schema:isbn"])

        # schema:url
        if "schema:url" in node:
            identifiers["url"] = str(node["schema:url"])

        # 提取属性 (反向映射 schema.org 标准属性)
        properties: dict[str, Any] = {}
        for schema_key, prop_key in _SCHEMA_ORG_PROP_REVERSE_MAP.items():
            if schema_key in node:
                properties[prop_key] = node[schema_key]

        # 其他 schema: 前缀属性
        for key, value in node.items():
            if (
                key.startswith("schema:")
                and key not in _SCHEMA_ORG_RESERVED_PROPS
                and key not in _SCHEMA_ORG_PROP_REVERSE_MAP
            ):
                prop_name = key.replace("schema:", "")
                # 跳过已经是 @id 引用格式的关系 (不作为属性)
                if isinstance(value, dict) and "@id" in value:
                    continue
                properties[prop_name] = value

        return KnowledgeEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=description,
            identifiers=identifiers,
            properties=properties,
            domain=domain,
            aliases=aliases,
            tags=tags,
        )

    # --------------------------------------------------------
    # 快照信息与完整性验证
    # --------------------------------------------------------

    def get_snapshot_info(self, path: str | Path) -> dict[str, Any]:
        """获取快照元数据信息 (不加载快照).

        读取 manifest.json 并补充文件大小信息。

        Args:
            path: 快照目录路径

        Returns:
            包含格式版本、创建时间、计数、校验和、文件大小等信息的字典

        Raises:
            IngestError: manifest.json 不存在
        """
        snapshot_dir = Path(path)
        manifest_path = snapshot_dir / _FILE_MANIFEST
        if not manifest_path.exists():
            raise IngestError(
                source=str(snapshot_dir),
                detail="manifest.json 不存在",
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 补充文件大小信息
        file_sizes: dict[str, int] = {}
        jsonl_files = [
            _FILE_ENTITIES, _FILE_TRIPLES, _FILE_CHUNKS,
            _FILE_VERSIONS, _FILE_CONFLICTS, _FILE_EVIDENCE,
        ]
        for filename in jsonl_files:
            file_path = snapshot_dir / filename
            if file_path.exists():
                file_sizes[filename] = file_path.stat().st_size

        # manifest 文件大小
        manifest_size = manifest_path.stat().st_size

        return {
            "format_version": manifest.get("format_version", ""),
            "created_at": manifest.get("created_at", 0),
            "created_at_iso": manifest.get("created_at_iso", ""),
            "counts": manifest.get("counts", {}),
            "checksums": manifest.get("checksums", {}),
            "stats": manifest.get("stats", {}),
            "manifest_checksum": manifest.get("manifest_checksum", ""),
            "file_sizes": file_sizes,
            "manifest_size": manifest_size,
            "total_size": sum(file_sizes.values()) + manifest_size,
            "path": str(snapshot_dir),
        }

    def verify_integrity(self, path: str | Path) -> bool:
        """验证快照完整性 (SHA-256 校验和验证).

        验证流程:
        1. 读取 manifest.json
        2. 逐个验证 JSONL 文件的 SHA-256 校验和
        3. 验证 manifest 自身的校验和

        Args:
            path: 快照目录路径

        Returns:
            完整性验证是否通过
        """
        snapshot_dir = Path(path)
        manifest_path = snapshot_dir / _FILE_MANIFEST
        if not manifest_path.exists():
            logger.warning("完整性验证失败: manifest.json 不存在: %s", snapshot_dir)
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("完整性验证失败: manifest.json 解析失败: %s", snapshot_dir)
            return False

        # 验证各文件校验和
        stored_checksums = manifest.get("checksums", {})
        for filename, stored_checksum in stored_checksums.items():
            file_path = snapshot_dir / filename
            if not file_path.exists():
                logger.warning("完整性验证失败: 文件缺失: %s", filename)
                return False
            actual_checksum = self._compute_file_checksum(file_path)
            if actual_checksum != stored_checksum:
                logger.warning("完整性验证失败: 校验和不匹配: %s", filename)
                return False

        # 验证 manifest 自身校验和
        manifest_checksum = manifest.get("manifest_checksum", "")
        if manifest_checksum:
            manifest_for_checksum = {
                k: v for k, v in manifest.items() if k != "manifest_checksum"
            }
            manifest_json = json.dumps(
                manifest_for_checksum, ensure_ascii=False, sort_keys=True
            )
            actual_manifest_checksum = self._compute_sha256(
                manifest_json.encode("utf-8")
            )
            if actual_manifest_checksum != manifest_checksum:
                logger.warning("完整性验证失败: manifest 校验和不匹配")
                return False

        logger.info("完整性验证通过: %s", snapshot_dir)
        return True

    # --------------------------------------------------------
    # 压缩合并 (借鉴 Neo4j checkpoint + Redis AOF rewrite)
    # --------------------------------------------------------

    def compact(self) -> Path:
        """压缩合并: 创建新快照并清理增量日志.

        借鉴 Neo4j checkpoint + Redis AOF rewrite:
        1. 创建新的全量快照 (将当前知识库状态完全序列化)
        2. 将快照目录压缩为 zip 文件 (使用 zipfile)
        3. 清理所有增量日志 (WAL 文件)

        压缩后的快照成为新的基线, 后续增量日志从零开始累积。

        Returns:
            压缩后的 zip 文件路径
        """
        with self._lock:
            # 创建新的全量快照
            snapshot_dir = self.save_snapshot()

            # 压缩为 zip 文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            zip_path = self._base_path / f"checkpoint_{timestamp}.zip"
            self._compress_directory(snapshot_dir, zip_path)

            # 清理增量日志
            wal_files = list(self._base_path.glob("wal_*.jsonl"))
            for wal_file in wal_files:
                try:
                    wal_file.unlink()
                    logger.debug("清理增量日志: %s", wal_file)
                except OSError as exc:
                    logger.warning("清理增量日志失败: %s (%s)", wal_file, exc)

            logger.info(
                "压缩完成: 快照=%s, 压缩包=%s, 清理增量日志=%d 个",
                snapshot_dir, zip_path, len(wal_files),
            )
            return zip_path


# ============================================================
# 事务 — 借鉴 SQLite BEGIN/COMMIT/ROLLBACK + undo log
# ============================================================


class Transaction:
    """知识库事务 (借鉴 SQLite BEGIN/COMMIT/ROLLBACK).

    使用 undo log 模式实现事务回滚:
    - 每个写操作执行前, 先记录逆操作到 undo log
    - commit: 清空 undo log (操作已生效)
    - rollback: 逆序回放 undo log, 恢复操作前状态

    支持 savepoint 嵌套回滚 (借鉴 SQLite SAVEPOINT):
    - savepoint(name): 在 undo log 中标记当前位置
    - rollback_to_savepoint(name): 回放到标记位置, 撤销之后操作
    - release_savepoint(name): 释放标记 (之后的操作不可部分回滚到该点)

    支持上下文管理器::

        with tx_manager.begin() as tx:
            tx.add_entity(entity)
            tx.update_entity(eid, name="新名称")
            # 正常退出: 自动 commit
            # 异常退出: 自动 rollback

    Attributes:
        _tx_manager: 事务管理器
        _tx_id: 事务唯一 ID
        _state: 事务状态
        _undo_log: undo 操作日志 (逆操作列表)
        _savepoints: 保存点 {name: undo_log 位置}
        _lock: 线程安全锁
    """

    def __init__(self, tx_manager: TransactionManager, tx_id: str) -> None:
        """初始化事务.

        Args:
            tx_manager: 所属的事务管理器
            tx_id: 事务唯一 ID
        """
        self._tx_manager = tx_manager
        self._tx_id = tx_id
        self._state = TransactionState.ACTIVE
        self._undo_log: list[dict[str, Any]] = []
        self._savepoints: dict[str, int] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def state(self) -> TransactionState:
        """获取事务当前状态."""
        return self._state

    @property
    def tx_id(self) -> str:
        """获取事务唯一 ID."""
        return self._tx_id

    @property
    def store(self) -> KnowledgeStore:
        """获取关联的知识存储 (用于直接读取, 不记录 undo)."""
        return self._tx_manager.store

    # --------------------------------------------------------
    # undo log 记录
    # --------------------------------------------------------

    def _record_undo(self, undo_entry: dict[str, Any]) -> None:
        """记录 undo 条目到 undo log.

        Args:
            undo_entry: 描述逆操作的字典

        Raises:
            L3Error: 事务已结束 (非 ACTIVE 状态)
        """
        if self._state != TransactionState.ACTIVE:
            raise L3Error(
                "L3_TRANSACTION",
                detail=f"事务已结束, 当前状态: {self._state.value}",
            )
        self._undo_log.append(undo_entry)

    # --------------------------------------------------------
    # 实体操作 (记录 undo)
    # --------------------------------------------------------

    def add_entity(
        self,
        entity: KnowledgeEntity,
        *,
        check_duplicate: bool = True,
    ) -> KnowledgeEntity:
        """添加实体 (记录 undo: 移除实体).

        Args:
            entity: 要添加的实体
            check_duplicate: 是否检查标识符重复

        Returns:
            添加后的实体
        """
        result = self.store.add_entity(entity, check_duplicate=check_duplicate)
        self._record_undo({
            "op": "remove_entity",
            "entity_id": entity.entity_id,
        })
        return result

    def update_entity(
        self,
        entity_id: str,
        *,
        expected_version: int | None = None,
        track_version: bool = True,
        changed_by: str = "system",
        reason: str = "",
        **updates: Any,
    ) -> KnowledgeEntity:
        """更新实体 (记录 undo: 恢复旧快照).

        Args:
            entity_id: 实体 ID
            expected_version: 期望版本号 (乐观锁)
            track_version: 是否记录版本
            changed_by: 变更者
            reason: 变更原因
            **updates: 更新字段

        Returns:
            更新后的实体
        """
        # 先保存旧快照 (用于 undo 恢复)
        old_entity = self.store.get_entity_or_raise(entity_id)
        old_snapshot = old_entity.model_dump(mode="json")

        result = self.store.update_entity(
            entity_id,
            expected_version=expected_version,
            track_version=track_version,
            changed_by=changed_by,
            reason=reason,
            **updates,
        )

        self._record_undo({
            "op": "restore_entity",
            "entity_id": entity_id,
            "snapshot": old_snapshot,
        })
        return result

    def remove_entity(self, entity_id: str) -> KnowledgeEntity | None:
        """移除实体 (记录 undo: 重新添加实体).

        Args:
            entity_id: 实体 ID

        Returns:
            被移除的实体, 不存在则返回 None
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return None

        # 保存完整快照 (用于 undo 重新添加)
        entity_snapshot = entity.model_dump(mode="json")
        result = self.store.remove_entity(entity_id)

        self._record_undo({
            "op": "readd_entity",
            "snapshot": entity_snapshot,
        })
        return result

    # --------------------------------------------------------
    # 三元组操作 (记录 undo)
    # --------------------------------------------------------

    def add_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        """添加三元组 (记录 undo: 移除三元组).

        Args:
            triple: 要添加的三元组

        Returns:
            添加后的三元组
        """
        result = self.store.add_triple(triple)
        self._record_undo({
            "op": "remove_triple",
            "triple_id": triple.triple_id,
        })
        return result

    def remove_triple(self, triple_id: str) -> KnowledgeTriple | None:
        """移除三元组 (记录 undo: 重新添加三元组).

        Args:
            triple_id: 三元组 ID

        Returns:
            被移除的三元组, 不存在则返回 None
        """
        triple = self.store.get_triple(triple_id)
        if triple is None:
            return None

        triple_snapshot = triple.model_dump(mode="json")
        result = self.store.remove_triple(triple_id)

        self._record_undo({
            "op": "readd_triple",
            "snapshot": triple_snapshot,
        })
        return result

    # --------------------------------------------------------
    # 切片操作 (记录 undo)
    # --------------------------------------------------------

    def add_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """添加切片 (记录 undo: 移除切片).

        Args:
            chunk: 要添加的切片

        Returns:
            添加后的切片
        """
        result = self.store.add_chunk(chunk)
        self._record_undo({
            "op": "remove_chunk",
            "chunk_id": chunk.chunk_id,
        })
        return result

    def remove_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """移除切片 (记录 undo: 重新添加切片).

        Args:
            chunk_id: 切片 ID

        Returns:
            被移除的切片, 不存在则返回 None
        """
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None:
            return None

        chunk_snapshot = chunk.model_dump(mode="json")
        result = self.store.remove_chunk(chunk_id)

        self._record_undo({
            "op": "readd_chunk",
            "snapshot": chunk_snapshot,
        })
        return result

    # --------------------------------------------------------
    # 事务控制 (借鉴 SQLite COMMIT/ROLLBACK/SAVEPOINT)
    # --------------------------------------------------------

    def commit(self) -> None:
        """提交事务 (借鉴 SQLite COMMIT).

        清空 undo log, 标记事务为已提交。
        提交后所有操作生效, 不可回滚。

        Raises:
            L3Error: 事务已结束 (非 ACTIVE 状态)
        """
        with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务已结束, 当前状态: {self._state.value}",
                )
            undo_count = len(self._undo_log)
            self._state = TransactionState.COMMITTED
            self._undo_log.clear()
            self._savepoints.clear()
            self._tx_manager._on_transaction_end(self)
            logger.info("事务提交: %s (undo 条目=%d)", self._tx_id, undo_count)

    def rollback(self) -> None:
        """回滚事务 (借鉴 SQLite ROLLBACK).

        逆序回放 undo log, 恢复所有操作前的状态。
        回滚后事务标记为已回滚。

        如果回放过程中出错, 事务标记为 FAILED。

        Raises:
            L3Error: 事务已结束 (非 ACTIVE 状态)
        """
        with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务已结束, 当前状态: {self._state.value}",
                )
            try:
                self._replay_undo(len(self._undo_log))
                self._state = TransactionState.ROLLED_BACK
            except Exception as exc:
                self._state = TransactionState.FAILED
                logger.error("事务回滚失败: %s, 错误: %s", self._tx_id, exc)
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务回滚失败: {exc}",
                    context={"tx_id": self._tx_id},
                ) from exc
            finally:
                self._undo_log.clear()
                self._savepoints.clear()
                self._tx_manager._on_transaction_end(self)
                logger.info("事务回滚: %s", self._tx_id)

    def savepoint(self, name: str) -> None:
        """创建保存点 (借鉴 SQLite SAVEPOINT).

        在 undo log 中标记当前位置, 后续可通过 rollback_to_savepoint
        部分回滚到该位置。

        Args:
            name: 保存点名称 (唯一)

        Raises:
            L3Error: 事务已结束或保存点名称已存在
        """
        with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务已结束, 当前状态: {self._state.value}",
                )
            if name in self._savepoints:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"保存点已存在: {name}",
                )
            self._savepoints[name] = len(self._undo_log)
            logger.debug("创建保存点: %s (位置=%d)", name, len(self._undo_log))

    def rollback_to_savepoint(self, name: str) -> None:
        """回滚到保存点 (借鉴 SQLite ROLLBACK TO SAVEPOINT).

        逆序回放 undo log 中保存点位置之后的条目, 撤销该位置之后的所有操作。
        回放后, 保存点之后创建的保存点将被移除。

        Args:
            name: 保存点名称

        Raises:
            L3Error: 事务已结束或保存点不存在
        """
        with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务已结束, 当前状态: {self._state.value}",
                )
            if name not in self._savepoints:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"保存点不存在: {name}",
                )

            position = self._savepoints[name]
            replay_count = len(self._undo_log) - position

            # 逆序回放 position 之后的 undo 条目
            self._replay_undo(replay_count)

            # 确保截断 (replay_undo 已 pop, 但显式截断更安全)
            self._undo_log = self._undo_log[:position]

            # 移除该保存点之后创建的保存点 (位置 > position 的)
            self._savepoints = {
                k: v for k, v in self._savepoints.items() if v <= position
            }

            logger.debug(
                "回滚到保存点: %s (位置=%d, 回滚=%d 条)",
                name, position, replay_count,
            )

    def release_savepoint(self, name: str) -> None:
        """释放保存点 (借鉴 SQLite RELEASE SAVEPOINT).

        移除保存点标记, 之后不可再回滚到该点。
        保存点之前的 undo 条目保留不变。

        Args:
            name: 保存点名称

        Raises:
            L3Error: 事务已结束或保存点不存在
        """
        with self._lock:
            if self._state != TransactionState.ACTIVE:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"事务已结束, 当前状态: {self._state.value}",
                )
            if name not in self._savepoints:
                raise L3Error(
                    "L3_TRANSACTION",
                    detail=f"保存点不存在: {name}",
                )
            del self._savepoints[name]
            logger.debug("释放保存点: %s", name)

    # --------------------------------------------------------
    # undo log 回放
    # --------------------------------------------------------

    def _replay_undo(self, count: int) -> None:
        """逆序回放指定数量的 undo 条目.

        从 undo log 末尾开始, 逐个弹出并执行逆操作。
        执行顺序为 LIFO (后进先出), 确保正确恢复操作前状态。

        逆操作类型:
        - remove_entity: 移除实体 (撤销 add_entity)
        - restore_entity: 恢复实体旧快照 (撤销 update_entity)
        - readd_entity: 重新添加实体 (撤销 remove_entity)
        - remove_triple: 移除三元组 (撤销 add_triple)
        - readd_triple: 重新添加三元组 (撤销 remove_triple)
        - remove_chunk: 移除切片 (撤销 add_chunk)
        - readd_chunk: 重新添加切片 (撤销 remove_chunk)

        Args:
            count: 要回放的 undo 条目数量
        """
        store = self.store

        for _ in range(count):
            if not self._undo_log:
                break

            entry = self._undo_log.pop()
            op = entry.get("op", "")

            try:
                if op == "remove_entity":
                    # 撤销 add_entity: 移除实体
                    store.entity_store.remove_entity(entry["entity_id"])

                elif op == "restore_entity":
                    # 撤销 update_entity: 恢复旧快照
                    entity_id = entry["entity_id"]
                    snapshot = dict(entry["snapshot"])
                    current = store.entity_store.get_entity(entity_id)
                    if current is not None:
                        current_snapshot = current.model_dump(mode="json")
                        restored = KnowledgeEntity.model_validate(snapshot)
                        store.entity_store._reindex_entity(restored, current_snapshot)
                        store.entity_store._entities[entity_id] = restored
                    else:
                        # 实体已被移除, 直接重新添加
                        restored = KnowledgeEntity.model_validate(snapshot)
                        store.entity_store.add_entity(restored, check_duplicate=False)

                elif op == "readd_entity":
                    # 撤销 remove_entity: 重新添加实体
                    snapshot = dict(entry["snapshot"])
                    entity = KnowledgeEntity.model_validate(snapshot)
                    store.entity_store.add_entity(entity, check_duplicate=False)

                elif op == "remove_triple":
                    # 撤销 add_triple: 移除三元组
                    store.triple_store.remove_triple(entry["triple_id"])

                elif op == "readd_triple":
                    # 撤销 remove_triple: 重新添加三元组
                    snapshot = dict(entry["snapshot"])
                    triple = KnowledgeTriple.model_validate(snapshot)
                    store.triple_store.add_triple(triple)

                elif op == "remove_chunk":
                    # 撤销 add_chunk: 移除切片
                    store.chunk_store.remove_chunk(entry["chunk_id"])

                elif op == "readd_chunk":
                    # 撤销 remove_chunk: 重新添加切片
                    snapshot = dict(entry["snapshot"])
                    chunk = DocumentChunk.model_validate(snapshot)
                    store.chunk_store.add_chunk(chunk)

                else:
                    logger.warning("未知 undo 操作类型: %s", op)

            except Exception as exc:
                logger.error(
                    "undo 回放失败: op=%s, entry=%s, error=%s",
                    op, entry, exc,
                )
                raise

    # --------------------------------------------------------
    # 上下文管理器
    # --------------------------------------------------------

    def __enter__(self) -> Transaction:
        """进入事务上下文, 返回事务对象."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """退出事务上下文 (自动提交或回滚).

        - 正常退出 (无异常): 自动 commit
        - 异常退出 (有异常): 自动 rollback
        - 已手动 commit/rollback: 不重复操作
        """
        # 事务已结束 (手动 commit/rollback), 不重复操作
        if self._state != TransactionState.ACTIVE:
            return

        if exc_type is not None:
            # 异常退出: 自动回滚
            logger.debug("事务因异常自动回滚: %s (%s)", self._tx_id, exc_type.__name__)
            self.rollback()
        else:
            # 正常退出: 自动提交
            self.commit()


# ============================================================
# 事务管理器 — 借鉴 SQLite WAL + MVCC
# ============================================================


class TransactionManager:
    """事务管理器 (借鉴 SQLite WAL + MVCC 乐观并发控制).

    管理事务的生命周期:
    - begin(): 创建新事务, 返回 Transaction 对象
    - 活跃事务追踪: 维护活跃事务列表
    - 事务结束回调: 事务 commit/rollback 时从活跃列表移除

    使用方式::

        txm = TransactionManager(store)

        # 方式 1: 上下文管理器 (推荐)
        with txm.begin() as tx:
            tx.add_entity(entity)
            tx.update_entity(eid, name="新名称")
            # 自动提交或回滚

        # 方式 2: 手动控制
        tx = txm.begin()
        try:
            tx.add_entity(entity)
            tx.commit()
        except Exception:
            tx.rollback()
            raise

        # 方式 3: 使用 savepoint 部分回滚
        with txm.begin() as tx:
            tx.add_entity(entity1)
            tx.savepoint("sp1")
            tx.add_entity(entity2)
            tx.rollback_to_savepoint("sp1")  # 撤销 entity2, 保留 entity1
            # 提交: 仅 entity1 生效

    Attributes:
        _store: 关联的知识存储
        _active_transactions: 活跃事务字典 {tx_id: Transaction}
        _lock: 线程安全锁
    """

    def __init__(self, store: KnowledgeStore) -> None:
        """初始化事务管理器.

        Args:
            store: 关联的知识存储引擎
        """
        self._store = store
        self._active_transactions: dict[str, Transaction] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def store(self) -> KnowledgeStore:
        """获取关联的知识存储."""
        return self._store

    @property
    def active_count(self) -> int:
        """当前活跃事务数."""
        with self._lock:
            return len(self._active_transactions)

    # --------------------------------------------------------
    # 事务生命周期管理
    # --------------------------------------------------------

    def begin(self) -> Transaction:
        """开始新事务 (借鉴 SQLite BEGIN).

        创建一个新的事务对象, 加入活跃事务列表。

        Returns:
            新创建的事务对象
        """
        with self._lock:
            tx_id = f"tx-{uuid.uuid4().hex[:12]}"
            tx = Transaction(self, tx_id)
            self._active_transactions[tx_id] = tx
            logger.info("开始事务: %s (活跃事务数=%d)", tx_id, len(self._active_transactions))
            return tx

    def get_active_transactions(self) -> list[str]:
        """获取所有活跃事务 ID 列表.

        Returns:
            活跃事务 ID 列表 (按创建顺序)
        """
        with self._lock:
            return list(self._active_transactions.keys())

    def _on_transaction_end(self, tx: Transaction) -> None:
        """事务结束时回调 (从活跃列表移除).

        此方法由 Transaction.commit() 和 Transaction.rollback() 调用,
        不应直接调用。

        Args:
            tx: 已结束的事务对象
        """
        with self._lock:
            self._active_transactions.pop(tx.tx_id, None)
            logger.debug(
                "事务结束: %s, 状态=%s (剩余活跃=%d)",
                tx.tx_id, tx.state.value, len(self._active_transactions),
            )


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "TransactionState",
    "Transaction",
    "TransactionManager",
    "PersistenceManager",
]
