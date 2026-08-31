"""外部开放数据源加载器 — 将临时外部知识缓存加载到 KnowledgeStore.

数据来源 (详见归档根目录 ``05-外部数据源/README-数据源说明.md``):
- Periodic-Table-JSON (MIT): 元素周期表基础物化数据
- Wikipedia Dysprosium 综述 (CC BY-SA 4.0)
- PubChem Dy2O3 化合物属性 (公共数据)

所有条目统一标记 ``KnowledgeSource(source_id=...)`` 与
``metadata["source"]="external"``, 与人工审核的正式知识严格区分。
加载采用确定性 ID, 幂等可重复执行。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .models import (
    AccessLevel,
    ChunkingStrategy,
    DocumentChunk,
    EntityType,
    KnowledgeEntity,
    KnowledgeSource,
    SourceTier,
)
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# 归档根目录下的外部数据缓存目录 (可被环境变量覆盖)
_ARCHIVE_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_DATA_DIR = Path(
    __import__("os").environ.get(
        "DY3_EXTERNAL_DATA_DIR",
        str(_ARCHIVE_ROOT / "05-外部数据源"),
    )
)

ACCESS_DATE = "2026-08-05"


def _source(source_id: str, name: str, endpoint: str, reliability: float = 0.85) -> KnowledgeSource:
    """构造统一的外部数据源元数据."""
    return KnowledgeSource(
        source_id=source_id,
        name=name,
        tier=SourceTier.TIER1_PUBLIC,
        endpoint=endpoint,
        auth_required=False,
        access_level=AccessLevel.PUBLIC,
        reliability=reliability,
        last_synced=time.time(),
        metadata={"external": True, "access_date": ACCESS_DATE},
    )


def _read_json(filename: str) -> dict[str, Any] | None:
    """读取外部缓存 JSON 文件, 缺失或损坏时返回 None."""
    path = EXTERNAL_DATA_DIR / filename
    if not path.exists():
        logger.warning("外部数据文件缺失: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("外部数据文件解析失败: %s (%s)", path, exc)
        return None


def load_periodic_table(store: KnowledgeStore) -> dict[str, int]:
    """加载元素周期表数据 (含全部镧系元素 + Sc/Y + Dy 重点条目)."""
    data = _read_json("periodic-table.json")
    if not data:
        return {"entities": 0, "chunks": 0}

    elements = data.get("elements", [])
    count = 0
    chunk_count = 0
    for item in elements:
        symbol = str(item.get("symbol", ""))
        number = int(item.get("number", 0) or 0)
        category = str(item.get("category", ""))
        is_rare_earth = category == "lanthanide" or symbol in {"Sc", "Y"}
        if not is_rare_earth and symbol != "Dy":
            continue

        entity_id = f"ext-periodic-{symbol.lower()}"
        if store.get_entity(entity_id) is not None:
            continue

        entity = KnowledgeEntity(
            entity_id=entity_id,
            entity_type=EntityType.MATERIAL,
            name=str(item.get("name", symbol)),
            description=str(item.get("summary", "")),
            identifiers={
                "atomic_number": str(number),
                "symbol": symbol,
            },
            properties={
                "atomic_mass": item.get("atomic_mass"),
                "category": category,
                "group": item.get("group"),
                "period": item.get("period"),
                "melting_point": item.get("melting_point"),
                "boiling_point": item.get("boiling_point"),
                "density": item.get("density"),
                "electron_configuration": item.get("electron_configuration"),
                "electron_configuration_semantic": item.get(
                    "electron_configuration_semantic"
                ),
                "discovered_by": item.get("discovered_by"),
                "appearance": item.get("appearance"),
            },
            domain="materials_science",
            tags=["稀土元素", "元素周期表", "物理化学基础"],
            aliases=[symbol],
            language="en",
            source=_source(
                "periodic_table_json",
                "Periodic-Table-JSON (Bowserinator)",
                "https://github.com/Bowserinator/Periodic-Table-JSON",
            ),
            metadata={"external": True, "access_date": ACCESS_DATE},
        )
        store.add_entity(entity)
        count += 1

        chunk_id = f"{entity_id}-chunk-001"
        if store.get_chunk(chunk_id) is None:
            content = (
                f"{entity.name} (symbol {symbol}, atomic number {number}): "
                f"{entity.description or ''} "
                f"Atomic mass {item.get('atomic_mass')}, "
                f"category {category}, melting point {item.get('melting_point')}, "
                f"boiling point {item.get('boiling_point')}, "
                f"density {item.get('density')}, "
                f"electron configuration {item.get('electron_configuration_semantic')}."
            )
            store.add_chunk(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=entity_id,
                    content=content,
                    section="periodic-table",
                    strategy=ChunkingStrategy.SEMANTIC_PARAGRAPH,
                    language="en",
                    metadata={
                        "source": "external",
                        "external_source": "periodic_table_json",
                        "source_url": "https://github.com/Bowserinator/Periodic-Table-JSON",
                        "access_date": ACCESS_DATE,
                    },
                )
            )
            chunk_count += 1

    return {"entities": count, "chunks": chunk_count}


def load_wikipedia_dysprosium(store: KnowledgeStore) -> dict[str, int]:
    """加载 Wikipedia Dysprosium 综述全文切片."""
    data = _read_json("wikipedia-dysprosium.json")
    if not data:
        return {"entities": 0, "chunks": 0}

    pages = data.get("query", {}).get("pages", {})
    extract = ""
    for page in pages.values():
        extract = page.get("extract", "")
        break
    if not extract:
        return {"entities": 0, "chunks": 0}

    entity_id = "ext-wiki-dysprosium"
    entity = None
    if store.get_entity(entity_id) is None:
        entity = KnowledgeEntity(
            entity_id=entity_id,
            entity_type=EntityType.CONCEPT,
            name="Dysprosium (Wikipedia 综述)",
            description=extract[:1000],
            identifiers={"wikipedia_title": "Dysprosium"},
            properties={"source_language": "en"},
            domain="materials_science",
            tags=["稀土元素", "镝", "基础知识"],
            aliases=["Dysprosium", "Dy"],
            language="en",
            source=_source(
                "wikipedia",
                "Wikipedia",
                "https://en.wikipedia.org/wiki/Dysprosium",
            ),
            metadata={"external": True, "access_date": ACCESS_DATE},
        )
        store.add_entity(entity)

    paragraphs = [p.strip() for p in extract.split("\n") if len(p.strip()) > 80]
    chunk_count = 0
    for index, para in enumerate(paragraphs, start=1):
        chunk_id = f"{entity_id}-chunk-{index:03d}"
        if store.get_chunk(chunk_id) is None:
            store.add_chunk(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=entity_id,
                    content=para,
                    section="wikipedia-overview",
                    chunk_index=index,
                    strategy=ChunkingStrategy.SEMANTIC_PARAGRAPH,
                    language="en",
                    metadata={
                        "source": "external",
                        "external_source": "wikipedia",
                        "source_url": "https://en.wikipedia.org/wiki/Dysprosium",
                        "access_date": ACCESS_DATE,
                        "license": "CC BY-SA 4.0",
                    },
                )
            )
            chunk_count += 1

    return {"entities": 1 if entity is not None else 0, "chunks": chunk_count}


def load_pubchem_dy2o3(store: KnowledgeStore) -> dict[str, int]:
    """加载 PubChem Dy2O3 化合物属性."""
    data = _read_json("pubchem-Dy2O3.json")
    if not data:
        return {"entities": 0, "chunks": 0}

    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return {"entities": 0, "chunks": 0}

    prop = props[0]
    entity_id = "ext-pubchem-dy2o3"
    count = 0
    chunk_count = 0
    if store.get_entity(entity_id) is None:
        entity = KnowledgeEntity(
            entity_id=entity_id,
            entity_type=EntityType.CHEMICAL_COMPOUND,
            name="Dy2O3 (氧化镝)",
            description="Dysprosium(III) oxide, a rare-earth oxide used in "
            "phosphors, ceramics and optical materials.",
            identifiers={
                "pubchem_cid": str(prop.get("CID", "")),
                "molecular_formula": str(prop.get("MolecularFormula", "")),
            },
            properties={
                "molecular_weight": prop.get("MolecularWeight"),
                "iupac_name": prop.get("IUPACName"),
                "canonical_smiles": prop.get("CanonicalSMILES"),
            },
            domain="chemistry",
            tags=["稀土氧化物", "发光材料", "Dy3+"],
            aliases=["Dysprosium oxide", "Dysprosium(III) oxide"],
            language="en",
            source=_source(
                "pubchem",
                "PubChem",
                "https://pubchem.ncbi.nlm.nih.gov",
            ),
            metadata={"external": True, "access_date": ACCESS_DATE},
        )
        store.add_entity(entity)
        count += 1

        chunk_id = f"{entity_id}-chunk-001"
        if store.get_chunk(chunk_id) is None:
            content = (
                f"Dy2O3 (Dysprosium(III) oxide): molecular formula "
                f"{prop.get('MolecularFormula')}, molecular weight "
                f"{prop.get('MolecularWeight')}, IUPAC name "
                f"{prop.get('IUPACName')}, SMILES "
                f"{prop.get('CanonicalSMILES')}. "
                "Dysprosium oxide is a precursor for Dy3+-doped phosphors "
                "and optical ceramics."
            )
            store.add_chunk(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=entity_id,
                    content=content,
                    section="pubchem-properties",
                    strategy=ChunkingStrategy.SEMANTIC_PARAGRAPH,
                    language="en",
                    metadata={
                        "source": "external",
                        "external_source": "pubchem",
                        "source_url": "https://pubchem.ncbi.nlm.nih.gov",
                        "access_date": ACCESS_DATE,
                    },
                )
            )
            chunk_count += 1

    return {"entities": count, "chunks": chunk_count}


def load_external_sources(store: KnowledgeStore) -> dict[str, Any]:
    """加载全部外部开放数据源 (幂等)."""
    summary: dict[str, Any] = {"loaded": False, "sources": {}, "total": {}}
    if not EXTERNAL_DATA_DIR.exists():
        logger.warning("外部数据目录不存在: %s", EXTERNAL_DATA_DIR)
        return summary

    results = {
        "periodic_table": load_periodic_table(store),
        "wikipedia_dysprosium": load_wikipedia_dysprosium(store),
        "pubchem_dy2o3": load_pubchem_dy2o3(store),
    }
    summary["loaded"] = any(r["entities"] or r["chunks"] for r in results.values())
    summary["sources"] = results
    summary["total"] = {
        "entities": sum(r["entities"] for r in results.values()),
        "chunks": sum(r["chunks"] for r in results.values()),
    }
    logger.info(
        "外部知识源加载完成: 实体=%d, 切片=%d",
        summary["total"]["entities"],
        summary["total"]["chunks"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = KnowledgeStore()
    result = load_external_sources(store)
    print(json.dumps(result, ensure_ascii=False, indent=2))
