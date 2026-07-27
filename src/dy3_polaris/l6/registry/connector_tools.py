"""20 个 L3 领域知识连接器工具的 Schema 定义与 Stub 实现.

分类:
- Tier-1 公共数据连接器 (10): nist, mp, ss, cie, icdd, pubchem, crossref, arxiv, wiki, openalex
- Tier-2 行业数据连接器 (6): cas, wos, scifinder, reaxys, thermocalc, vasp
- Tier-3 校园数据连接器 (4): library, edu, lims, campus_kb

每个工具包含完整的 input_schema / output_schema / Dy3ToolAnnotations。
命名规范: 连接器工具使用下划线命名 (与 ToolRegistration pattern 一致)。
所有工具 layer=LayerTag.L3_DOMAIN_KNOWLEDGE，category 按层级区分。
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)

logger = logging.getLogger(__name__)


# ============================================================
# Tier-1 公共数据连接器 (10)
# ============================================================

def _nist_query_spectrum_registration() -> ToolRegistration:
    """nist_query_spectrum — NIST 化学光谱查询.

    endpoint: webbook.nist.gov, auth: none, rate_limit: 60
    """
    return ToolRegistration(
        name="nist_query_spectrum",
        description="NIST 化学光谱查询：通过 NIST Chemistry WebBook 检索化合物的红外(IR)、紫外可见(UV-Vis)、质谱(MS)等光谱数据。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "compound": {
                    "type": "string",
                    "description": "化合物名称、化学式或 CAS 号，如 'water' 或 '7732-18-5'",
                },
                "spectrum_type": {
                    "type": "string",
                    "enum": ["IR", "UV-Vis", "MS", "Raman"],
                    "default": "IR",
                    "description": "光谱类型",
                },
                "units": {
                    "type": "string",
                    "enum": ["cm-1", "nm", "m/z"],
                    "default": "cm-1",
                    "description": "横轴单位",
                },
                "max_peaks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 50,
                    "description": "返回峰值数量上限",
                },
            },
            "required": ["compound", "spectrum_type"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "compound": {"type": "string", "description": "查询的化合物"},
                "spectrum_type": {"type": "string"},
                "cas_number": {"type": "string", "description": "CAS 号"},
                "peaks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "number", "description": "峰位置 (按 units)"},
                            "intensity": {"type": "number", "description": "相对强度"},
                            "assignment": {"type": "string", "description": "峰归属"},
                        },
                    },
                    "description": "光谱峰值列表",
                },
                "source": {"type": "string", "description": "数据来源标识"},
            },
            "required": ["compound", "spectrum_type", "peaks", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["nist", "spectrum", "chemistry", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=800,
            domain_scope=["DOM-A", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _nist_query_spectrum_handler(
    compound: str,
    spectrum_type: str = "IR",
    units: str = "cm-1",
    max_peaks: int = 50,
) -> dict[str, Any]:
    """NIST 光谱查询 (stub: 返回模拟光谱数据)."""
    mock_cas = {"water": "7732-18-5", "ethanol": "64-17-5", "benzene": "71-43-2"}.get(
        compound.lower(), "00-00-0"
    )
    base_peaks: list[dict[str, Any]] = [
        {"position": 3400.0, "intensity": 0.95, "assignment": "O-H stretch"},
        {"position": 1640.0, "intensity": 0.60, "assignment": "H-O-H bend"},
        {"position": 2120.0, "intensity": 0.20, "assignment": "combination"},
    ]
    if spectrum_type == "MS":
        base_peaks = [
            {"position": 18.0, "intensity": 1.00, "assignment": "M+ (molecular ion)"},
            {"position": 17.0, "intensity": 0.45, "assignment": "M-1"},
            {"position": 16.0, "intensity": 0.18, "assignment": "O+"},
        ]
        units = "m/z"
    return {
        "compound": compound,
        "spectrum_type": spectrum_type,
        "cas_number": mock_cas,
        "peaks": base_peaks[:max_peaks],
        "source": "NIST Chemistry WebBook (webbook.nist.gov)",
    }


def _mp_query_material_registration() -> ToolRegistration:
    """mp_query_material — Materials Project 材料数据查询.

    endpoint: api.materialsproject.org, auth: API key, rate_limit: 100
    """
    return ToolRegistration(
        name="mp_query_material",
        description="Materials Project 材料数据查询：通过 MP API 检索材料的晶体结构、电子结构、热力学等计算性质。需 API key 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "化学式，如 'Fe2O3'，或 material_id 如 'mp-19770'",
                },
                "properties_requested": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["structure", "band_gap", "formation_energy_per_atom", "density", "elasticity", "magnetic"],
                    },
                    "default": ["structure", "band_gap", "formation_energy_per_atom"],
                    "description": "请求的属性列表",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "required": ["formula"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
                "materials": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "material_id": {"type": "string"},
                            "formula_pretty": {"type": "string"},
                            "band_gap": {"type": "number"},
                            "formation_energy_per_atom": {"type": "number"},
                            "density": {"type": "number"},
                            "crystal_system": {"type": "string"},
                        },
                    },
                    "description": "匹配的材料列表",
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["formula", "materials", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["materials_project", "materials", "DFT", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=600,
            domain_scope=["DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _mp_query_material_handler(
    formula: str,
    properties_requested: list[str] | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Materials Project 材料查询 (stub)."""
    if properties_requested is None:
        properties_requested = ["structure", "band_gap", "formation_energy_per_atom"]
    mock_materials: list[dict[str, Any]] = [
        {
            "material_id": "mp-19770",
            "formula_pretty": formula,
            "band_gap": 2.1,
            "formation_energy_per_atom": -2.84,
            "density": 5.24,
            "crystal_system": "cubic",
        },
        {
            "material_id": "mp-790415",
            "formula_pretty": formula,
            "band_gap": 0.0,
            "formation_energy_per_atom": -1.92,
            "density": 7.87,
            "crystal_system": "hexagonal",
        },
    ]
    return {
        "formula": formula,
        "materials": mock_materials[:max_results],
        "total_found": len(mock_materials),
        "source": "Materials Project (api.materialsproject.org)",
    }


def _ss_search_paper_registration() -> ToolRegistration:
    """ss_search_paper — Semantic Scholar 论文搜索.

    endpoint: api.semanticscholar.org, auth: optional API key, rate_limit: 100
    """
    return ToolRegistration(
        name="ss_search_paper",
        description="Semantic Scholar 论文搜索：基于语义检索学术论文，支持按关键词/短语匹配，返回标题、摘要、引用数、影响力等元数据。可选 API key 提升配额。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["title", "authors", "year", "abstract", "citationCount", "externalIds"],
                    "description": "请求返回的字段",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "year_range": {
                    "type": "string",
                    "description": "年份范围，如 '2020-2024'",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paperId": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "year": {"type": "integer"},
                            "abstract": {"type": "string"},
                            "citationCount": {"type": "integer"},
                            "doi": {"type": "string"},
                        },
                    },
                },
                "total": {"type": "integer"},
                "offset": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["query", "papers", "total", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["semantic_scholar", "paper", "literature", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=700,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _ss_search_paper_handler(
    query: str,
    fields: list[str] | None = None,
    limit: int = 10,
    offset: int = 0,
    year_range: str = "",
) -> dict[str, Any]:
    """Semantic Scholar 论文搜索 (stub)."""
    mock_papers: list[dict[str, Any]] = [
        {
            "paperId": "abc123def456",
            "title": f"Advances in {query}: a comprehensive review",
            "authors": ["J. Smith", "L. Wang"],
            "year": 2023,
            "abstract": f"This review surveys recent progress in {query}...",
            "citationCount": 142,
            "doi": "10.1000/mock.2023.001",
        },
        {
            "paperId": "xyz789ghi012",
            "title": f"Experimental study of {query} under extreme conditions",
            "authors": ["M. Garcia", "T. Chen", "R. Patel"],
            "year": 2022,
            "abstract": f"We report measurements of {query}...",
            "citationCount": 67,
            "doi": "10.1000/mock.2022.042",
        },
    ]
    return {
        "query": query,
        "papers": mock_papers[:limit],
        "total": len(mock_papers),
        "offset": offset,
        "source": "Semantic Scholar (api.semanticscholar.org)",
    }


def _cie_get_colorimetry_registration() -> ToolRegistration:
    """cie_get_colorimetry — CIE 色度学数据查询.

    endpoint: cie.co.at, auth: none, rate_limit: 30
    """
    return ToolRegistration(
        name="cie_get_colorimetry",
        description="CIE 色度学数据查询：获取 CIE 标准色度学数据，包括光谱三刺激值(色匹配函数)、标准光源相对功率分布、色度坐标等。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["color_matching_function", "illuminant", "chromaticity"],
                    "default": "color_matching_function",
                    "description": "数据类型",
                },
                "observer": {
                    "type": "string",
                    "enum": ["1931_2deg", "1964_10deg"],
                    "default": "1931_2deg",
                    "description": "CIE 标准观察者",
                },
                "illuminant": {
                    "type": "string",
                    "enum": ["A", "D65", "D50", "C"],
                    "default": "D65",
                    "description": "标准光源(仅 illuminant/chromaticity 类型有效)",
                },
                "wavelength_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number", "minimum": 380, "maximum": 780},
                        "end": {"type": "number", "minimum": 380, "maximum": 780},
                        "step": {"type": "number", "enum": [1, 5, 10]},
                    },
                    "description": "波长范围(nm)",
                },
            },
            "required": ["data_type"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "data_type": {"type": "string"},
                "observer": {"type": "string"},
                "illuminant": {"type": "string"},
                "data_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "wavelength_nm": {"type": "number"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                    },
                    "description": "色度学数据点",
                },
                "white_point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "description": "白点色坐标",
                },
                "source": {"type": "string"},
            },
            "required": ["data_type", "data_points", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["cie", "colorimetry", "optics", "color", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=500,
            domain_scope=["DOM-C"],
            rate_limit=30,
        ),
    )


async def _cie_get_colorimetry_handler(
    data_type: str = "color_matching_function",
    observer: str = "1931_2deg",
    illuminant: str = "D65",
    wavelength_range: dict | None = None,
) -> dict[str, Any]:
    """CIE 色度学数据查询 (stub)."""
    if wavelength_range is None:
        wavelength_range = {"start": 380, "end": 780, "step": 10}
    start = wavelength_range.get("start", 380)
    end = wavelength_range.get("end", 780)
    step = wavelength_range.get("step", 10)
    # 简化的 CIE 1931 色匹配函数近似(钟形)
    import math as _math
    data_points: list[dict[str, Any]] = []
    wl = start
    while wl <= end:
        x = 1.056 * _math.exp(-((wl - 599.8) ** 2) / (2 * 37.0 ** 2)) \
            + 0.362 * _math.exp(-((wl - 442.0) ** 2) / (2 * 16.0 ** 2))
        y = 1.011 * _math.exp(-((wl - 556.3) ** 2) / (2 * 40.5 ** 2))
        z = 2.060 * _math.exp(-((wl - 449.8) ** 2) / (2 * 22.5 ** 2))
        data_points.append({"wavelength_nm": wl, "x": round(x, 4), "y": round(y, 4), "z": round(z, 4)})
        wl += step
    white_points = {"D65": {"x": 0.3127, "y": 0.3290}, "D50": {"x": 0.3457, "y": 0.3585},
                    "A": {"x": 0.4476, "y": 0.4074}, "C": {"x": 0.3101, "y": 0.3162}}
    return {
        "data_type": data_type,
        "observer": observer,
        "illuminant": illuminant,
        "data_points": data_points,
        "white_point": white_points.get(illuminant, {"x": 0.3127, "y": 0.3290}),
        "source": "CIE (cie.co.at)",
    }


def _icdd_xrd_match_registration() -> ToolRegistration:
    """icdd_xrd_match — ICDD XRD 衍射数据匹配.

    endpoint: icdd.com, auth: subscription, rate_limit: 20
    """
    return ToolRegistration(
        name="icdd_xrd_match",
        description="ICDD XRD 衍射数据匹配：将实测 X 射线衍射 2θ 峰值与 ICDD PDF 数据库进行匹配，鉴定物相组成。需订阅认证。",
        input_schema={
            "type": "object",
            "properties": {
                "peaks_2theta": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "two_theta": {"type": "number", "minimum": 0, "maximum": 180},
                            "intensity": {"type": "number", "minimum": 0},
                        },
                        "required": ["two_theta"],
                    },
                    "minItems": 1,
                    "description": "实测衍射峰值列表",
                },
                "wavelength": {
                    "type": "number",
                    "default": 1.5406,
                    "description": "X 射线波长 (Å)，Cu Kα 默认 1.5406",
                },
                "max_phases": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "最多返回匹配相数",
                },
            },
            "required": ["peaks_2theta"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "matched_phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pdf_number": {"type": "string", "description": "ICDD PDF 卡片号"},
                            "compound": {"type": "string"},
                            "formula": {"type": "string"},
                            "match_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "crystal_system": {"type": "string"},
                            "reference_peaks": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "参考 2θ 峰位",
                            },
                        },
                    },
                    "description": "匹配的物相",
                },
                "wavelength": {"type": "number"},
                "best_match": {"type": "string", "description": "最佳匹配 PDF 卡片号"},
                "source": {"type": "string"},
            },
            "required": ["matched_phases", "wavelength", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["icdd", "xrd", "diffraction", "phase_identification", "L3", "connector"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=1200,
            domain_scope=["DOM-B", "DOM-C"],
            rate_limit=20,
        ),
    )


async def _icdd_xrd_match_handler(
    peaks_2theta: list[dict],
    wavelength: float = 1.5406,
    max_phases: int = 3,
) -> dict[str, Any]:
    """ICDD XRD 匹配 (stub)."""
    mock_phases: list[dict[str, Any]] = [
        {
            "pdf_number": "00-021-1272",
            "compound": "SiO2 (α-Quartz)",
            "formula": "SiO2",
            "match_score": 0.92,
            "crystal_system": "hexagonal",
            "reference_peaks": [20.86, 26.64, 36.54, 39.47, 50.13],
        },
        {
            "pdf_number": "00-005-0626",
            "compound": "Iron Oxide (Hematite)",
            "formula": "Fe2O3",
            "match_score": 0.74,
            "crystal_system": "trigonal",
            "reference_peaks": [24.14, 33.15, 35.61, 40.85, 49.48],
        },
    ]
    return {
        "matched_phases": mock_phases[:max_phases],
        "wavelength": wavelength,
        "best_match": mock_phases[0]["pdf_number"] if mock_phases else None,
        "source": "ICDD PDF Database (icdd.com)",
    }


def _pubchem_query_compound_registration() -> ToolRegistration:
    """pubchem_query_compound — PubChem 化合物查询.

    endpoint: pubchem.ncbi.nlm.nih.gov, auth: none, rate_limit: 200
    """
    return ToolRegistration(
        name="pubchem_query_compound",
        description="PubChem 化合物查询：通过化合物名称、CID、SMILES 或 InChI 检索 PubChem 化合物信息，包括分子式、分子量、结构、IUPAC 名称等。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "化合物标识：名称/CID/SMILES/InChI，如 'aspirin' 或 'CC(=O)Oc1ccccc1C(=O)O'",
                },
                "identifier_type": {
                    "type": "string",
                    "enum": ["name", "cid", "smiles", "inchi"],
                    "default": "name",
                    "description": "标识类型",
                },
                "properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["MolecularFormula", "MolecularWeight", "CanonicalSMILES", "IUPACName"],
                    "description": "请求的属性",
                },
            },
            "required": ["identifier"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "integer", "description": "PubChem CID"},
                "identifier": {"type": "string"},
                "molecular_formula": {"type": "string"},
                "molecular_weight": {"type": "number"},
                "canonical_smiles": {"type": "string"},
                "iupac_name": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["cid", "identifier", "molecular_formula", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["pubchem", "compound", "chemistry", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=600,
            domain_scope=["DOM-A"],
            rate_limit=200,
        ),
    )


async def _pubchem_query_compound_handler(
    identifier: str,
    identifier_type: str = "name",
    properties: list[str] | None = None,
) -> dict[str, Any]:
    """PubChem 化合物查询 (stub)."""
    mock_db: dict[str, dict[str, Any]] = {
        "aspirin": {
            "cid": 2244,
            "molecular_formula": "C9H8O4",
            "molecular_weight": 180.16,
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "iupac_name": "2-acetyloxybenzoic acid",
        },
        "caffeine": {
            "cid": 2519,
            "molecular_formula": "C8H10N4O2",
            "molecular_weight": 194.19,
            "canonical_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "iupac_name": "1,3,7-trimethylpurine-2,6-dione",
        },
    }
    entry = mock_db.get(identifier.lower(), {
        "cid": 99999,
        "molecular_formula": "C6H12O6",
        "molecular_weight": 180.16,
        "canonical_smiles": "OCC(O)C(O)C(O)C(O)C=O",
        "iupac_name": "(2R,3S,4R,5R)-2,3,4,5,6-pentahydroxyhexanal",
    })
    return {
        "cid": entry["cid"],
        "identifier": identifier,
        "molecular_formula": entry["molecular_formula"],
        "molecular_weight": entry["molecular_weight"],
        "canonical_smiles": entry["canonical_smiles"],
        "iupac_name": entry["iupac_name"],
        "source": "PubChem (pubchem.ncbi.nlm.nih.gov)",
    }


def _crossref_resolve_doi_registration() -> ToolRegistration:
    """crossref_resolve_doi — Crossref DOI 解析.

    endpoint: api.crossref.org, auth: mailto, rate_limit: 50
    """
    return ToolRegistration(
        name="crossref_resolve_doi",
        description="Crossref DOI 解析：解析数字对象标识符(DOI)，获取文献的完整元数据，包括标题、作者、期刊、出版商、年份、ISSN 等。建议提供 mailto 进入友好池。",
        input_schema={
            "type": "object",
            "properties": {
                "doi": {
                    "type": "string",
                    "pattern": r"^10\.\d{4,9}/.+$",
                    "description": "DOI，如 '10.1038/nature12373'",
                },
                "mailto": {
                    "type": "string",
                    "description": "联系邮箱(用于 Crossref 友好池，提升速率)",
                },
            },
            "required": ["doi"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "doi": {"type": "string"},
                "title": {"type": "string"},
                "authors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "given": {"type": "string"},
                            "family": {"type": "string"},
                        },
                    },
                },
                "container_title": {"type": "string", "description": "期刊/来源名"},
                "publisher": {"type": "string"},
                "published_year": {"type": "integer"},
                "issn": {"type": "string"},
                "type": {"type": "string", "description": "文献类型"},
                "source": {"type": "string"},
            },
            "required": ["doi", "title", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["crossref", "doi", "metadata", "literature", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=600,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _crossref_resolve_doi_handler(
    doi: str,
    mailto: str = "",
) -> dict[str, Any]:
    """Crossref DOI 解析 (stub)."""
    return {
        "doi": doi,
        "title": "Resolved publication title (stub)",
        "authors": [
            {"given": "A.", "family": "Researcher"},
            {"given": "B.", "family": "Coauthor"},
        ],
        "container_title": "Journal of Stubbed Sciences",
        "publisher": "Example Publisher",
        "published_year": 2023,
        "issn": "1234-5678",
        "type": "journal-article",
        "source": "Crossref (api.crossref.org)",
    }


def _arxiv_search_paper_registration() -> ToolRegistration:
    """arxiv_search_paper — arXiv 论文搜索.

    endpoint: export.arxiv.org, auth: none, rate_limit: 60
    """
    return ToolRegistration(
        name="arxiv_search_paper",
        description="arXiv 论文搜索：检索 arXiv 预印本库，支持按关键词和学科分类搜索，返回 arXiv ID、标题、作者、摘要、分类等。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "category": {
                    "type": "string",
                    "description": "arXiv 分类，如 'cond-mat.mtrl-sci'、'physics.optics'",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "default": "relevance",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arxiv_id": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "abstract": {"type": "string"},
                            "categories": {"type": "array", "items": {"type": "string"}},
                            "published": {"type": "string"},
                            "pdf_url": {"type": "string"},
                        },
                    },
                },
                "total": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["query", "papers", "total", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["arxiv", "preprint", "paper", "physics", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=700,
            domain_scope=["DOM-B", "DOM-C"],
            rate_limit=60,
        ),
    )


async def _arxiv_search_paper_handler(
    query: str,
    category: str = "",
    max_results: int = 10,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    """arXiv 论文搜索 (stub)."""
    cats = [category] if category else ["cond-mat.mtrl-sci", "physics.optics"]
    mock_papers: list[dict[str, Any]] = [
        {
            "arxiv_id": "2310.00001",
            "title": f"Efficient methods for {query}",
            "authors": ["A. Physicist", "B. Theorist"],
            "abstract": f"In this work we present novel approaches to {query}...",
            "categories": cats,
            "published": "2023-10-01T00:00:00Z",
            "pdf_url": "https://arxiv.org/pdf/2310.00001",
        },
        {
            "arxiv_id": "2309.99999",
            "title": f"Observation of anomalous behavior in {query}",
            "authors": ["C. Experimentalist"],
            "abstract": f"We report experimental observations regarding {query}...",
            "categories": cats,
            "published": "2023-09-28T00:00:00Z",
            "pdf_url": "https://arxiv.org/pdf/2309.99999",
        },
    ]
    return {
        "query": query,
        "papers": mock_papers[:max_results],
        "total": len(mock_papers),
        "source": "arXiv (export.arxiv.org)",
    }


def _wiki_get_summary_registration() -> ToolRegistration:
    """wiki_get_summary — Wikipedia 摘要获取.

    endpoint: en.wikipedia.org, auth: none, rate_limit: 300
    """
    return ToolRegistration(
        name="wiki_get_summary",
        description="Wikipedia 摘要获取：获取维基百科条目的摘要文本、缩略图和页面 URL，支持多语言。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "条目标题，如 'Photoelectric effect'",
                },
                "language": {
                    "type": "string",
                    "default": "en",
                    "description": "语言代码，如 'en'、'zh'",
                },
                "sentences": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "摘要句数",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "extract": {"type": "string", "description": "摘要文本"},
                "url": {"type": "string", "description": "条目 URL"},
                "thumbnail": {"type": "string", "description": "缩略图 URL"},
                "language": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["title", "extract", "url", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["wikipedia", "encyclopedia", "summary", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=400,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=300,
        ),
    )


async def _wiki_get_summary_handler(
    title: str,
    language: str = "en",
    sentences: int = 5,
) -> dict[str, Any]:
    """Wikipedia 摘要获取 (stub)."""
    extract = (
        f"{title} is a concept described on Wikipedia. This is a stub summary "
        f"of the first {sentences} sentences for the '{title}' article. "
        "In production, this text is fetched from the MediaWiki REST API."
    )
    return {
        "title": title,
        "extract": extract,
        "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "thumbnail": f"https://upload.wikimedia.org/wikipedia/commons/thumb/mock/{title}.png",
        "language": language,
        "source": "Wikipedia (en.wikipedia.org)",
    }


def _openalex_search_works_registration() -> ToolRegistration:
    """openalex_search_works — OpenAlex 学术作品搜索.

    endpoint: api.openalex.org, auth: none, rate_limit: 100
    """
    return ToolRegistration(
        name="openalex_search_works",
        description="OpenAlex 学术作品搜索：检索 OpenAlex 开放学术图谱中的作品，返回标题、作者、年份、被引次数、DOI、概念标签等。无需认证。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "filter": {
                    "type": "string",
                    "description": "过滤条件，如 'from_publication_date:2020-01-01'",
                },
                "per_page": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
                "sort": {
                    "type": "string",
                    "enum": ["relevance_score", "cited_by_count", "publication_date"],
                    "default": "relevance_score",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "works": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "OpenAlex Work ID"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "publication_year": {"type": "integer"},
                            "cited_by_count": {"type": "integer"},
                            "doi": {"type": "string"},
                            "concepts": {"type": "array", "items": {"type": "string"}},
                            "open_access": {"type": "boolean"},
                        },
                    },
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "per_page": {"type": "integer"},
                    },
                },
                "source": {"type": "string"},
            },
            "required": ["query", "works", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["openalex", "academic", "works", "bibliography", "L3", "connector", "public_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER1,
            estimated_latency_ms=600,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _openalex_search_works_handler(
    query: str,
    filter: str = "",
    per_page: int = 25,
    sort: str = "relevance_score",
) -> dict[str, Any]:
    """OpenAlex 作品搜索 (stub)."""
    mock_works: list[dict[str, Any]] = [
        {
            "id": "W2741809807",
            "title": f"Open scholarly work on {query}",
            "authors": ["E. Scholar", "F. Author"],
            "publication_year": 2021,
            "cited_by_count": 89,
            "doi": "https://doi.org/10.1000/openalex.2021.001",
            "concepts": [query, "methodology", "analysis"],
            "open_access": True,
        },
        {
            "id": "W3097681234",
            "title": f"A data-driven approach to {query}",
            "authors": ["G. Researcher"],
            "publication_year": 2022,
            "cited_by_count": 215,
            "doi": "https://doi.org/10.1000/openalex.2022.042",
            "concepts": [query, "data science"],
            "open_access": False,
        },
    ]
    return {
        "query": query,
        "works": mock_works[:per_page],
        "meta": {"count": len(mock_works), "per_page": per_page},
        "source": "OpenAlex (api.openalex.org)",
    }


# ============================================================
# Tier-2 行业数据连接器 (6)
# ============================================================

def _cas_query_substance_registration() -> ToolRegistration:
    """cas_query_substance — CAS 化学物质查询.

    endpoint: commonchemistry.org, auth: subscription, rate_limit: 30
    """
    return ToolRegistration(
        name="cas_query_substance",
        description="CAS 化学物质查询：通过 CAS 登记号或名称检索化学物质信息，包括 CAS RN、分子式、分子量、物性数据等。需订阅认证。",
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "CAS RN (如 '64-17-5') 或物质名称",
                },
                "identifier_type": {
                    "type": "string",
                    "enum": ["cas_rn", "name"],
                    "default": "cas_rn",
                },
                "include_properties": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否包含物性数据",
                },
            },
            "required": ["identifier"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "cas_rn": {"type": "string"},
                "name": {"type": "string"},
                "molecular_formula": {"type": "string"},
                "molecular_weight": {"type": "number"},
                "properties": {
                    "type": "object",
                    "properties": {
                        "melting_point": {"type": "string"},
                        "boiling_point": {"type": "string"},
                        "density": {"type": "string"},
                        "solubility": {"type": "string"},
                    },
                    "description": "物性数据",
                },
                "source": {"type": "string"},
            },
            "required": ["cas_rn", "name", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["cas", "substance", "chemistry", "L3", "connector", "industry_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=900,
            domain_scope=["DOM-A"],
            rate_limit=30,
        ),
    )


async def _cas_query_substance_handler(
    identifier: str,
    identifier_type: str = "cas_rn",
    include_properties: bool = True,
) -> dict[str, Any]:
    """CAS 物质查询 (stub)."""
    mock_db: dict[str, dict[str, Any]] = {
        "64-17-5": {
            "cas_rn": "64-17-5",
            "name": "Ethanol",
            "molecular_formula": "C2H6O",
            "molecular_weight": 46.07,
            "properties": {
                "melting_point": "-114.14 °C",
                "boiling_point": "78.29 °C",
                "density": "0.789 g/cm³",
                "solubility": "Miscible with water",
            },
        },
    }
    entry = mock_db.get(identifier, {
        "cas_rn": identifier if identifier_type == "cas_rn" else "00-00-0",
        "name": identifier,
        "molecular_formula": "CxHyOz",
        "molecular_weight": 0.0,
        "properties": {
            "melting_point": "N/A",
            "boiling_point": "N/A",
            "density": "N/A",
            "solubility": "N/A",
        },
    })
    return {
        "cas_rn": entry["cas_rn"],
        "name": entry["name"],
        "molecular_formula": entry["molecular_formula"],
        "molecular_weight": entry["molecular_weight"],
        "properties": entry["properties"] if include_properties else {},
        "source": "CAS Common Chemistry (commonchemistry.org)",
    }


def _wos_search_article_registration() -> ToolRegistration:
    """wos_search_article — Web of Science 文章搜索.

    endpoint: clarivate.com, auth: API key, rate_limit: 25
    """
    return ToolRegistration(
        name="wos_search_article",
        description="Web of Science 文章搜索：检索 Web of Science 核心合集，支持按主题、作者、机构等检索，返回 UT、标题、作者、来源、被引次数、DOI 等。需 API key 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索式 (TS=/AU=/SO= 等)"},
                "database": {
                    "type": "string",
                    "enum": ["WOS", "SCI", "SSCI", "AHCI", "CPCI"],
                    "default": "WOS",
                    "description": "数据库标识",
                },
                "timespan": {
                    "type": "object",
                    "properties": {
                        "begin": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                        "end": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    },
                    "description": "时间范围",
                },
                "count": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ut": {"type": "string", "description": "WoS UT accession number"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "source": {"type": "string", "description": "来源期刊"},
                            "published_year": {"type": "integer"},
                            "times_cited": {"type": "integer"},
                            "doi": {"type": "string"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["query", "records", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["wos", "web_of_science", "citation", "L3", "connector", "industry_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=1000,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=25,
        ),
    )


async def _wos_search_article_handler(
    query: str,
    database: str = "WOS",
    timespan: dict | None = None,
    count: int = 25,
) -> dict[str, Any]:
    """Web of Science 文章搜索 (stub)."""
    mock_records: list[dict[str, Any]] = [
        {
            "ut": "WOS:000123456789",
            "title": f"Web of Science indexed article on {query}",
            "authors": ["H. Investigator", "I. Coauthor"],
            "source": "Nature Materials",
            "published_year": 2022,
            "times_cited": 312,
            "doi": "10.1038/nmat.2022.001",
        },
    ]
    return {
        "query": query,
        "records": mock_records[:count],
        "total_found": len(mock_records),
        "source": "Web of Science (clarivate.com)",
    }


def _scifinder_react_search_registration() -> ToolRegistration:
    """scifinder_react_search — SciFinder 反应搜索.

    endpoint: scifinder.cas.org, auth: subscription, rate_limit: 20
    """
    return ToolRegistration(
        name="scifinder_react_search",
        description="SciFinder 反应搜索：基于反应物/产物结构或名称检索化学反应，返回反应方案、条件、产率和文献来源。需订阅认证。",
        input_schema={
            "type": "object",
            "properties": {
                "reactant": {
                    "type": "string",
                    "description": "反应物名称/SMILES/CAS",
                },
                "product": {
                    "type": "string",
                    "description": "产物名称/SMILES/CAS",
                },
                "reaction_type": {
                    "type": "string",
                    "enum": ["synthesis", "substitution", "oxidation", "reduction", "catalysis", "any"],
                    "default": "any",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["reactant"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "reactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reaction_id": {"type": "string"},
                            "scheme": {"type": "string", "description": "反应方案文本描述"},
                            "reactants": {"type": "array", "items": {"type": "string"}},
                            "products": {"type": "array", "items": {"type": "string"}},
                            "conditions": {"type": "string", "description": "反应条件"},
                            "yield_percent": {"type": "number", "minimum": 0, "maximum": 100},
                            "reference": {"type": "string"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["reactions", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["scifinder", "reaction", "chemistry", "synthesis", "L3", "connector", "industry_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=1200,
            domain_scope=["DOM-A"],
            rate_limit=20,
        ),
    )


async def _scifinder_react_search_handler(
    reactant: str,
    product: str = "",
    reaction_type: str = "any",
    max_results: int = 10,
) -> dict[str, Any]:
    """SciFinder 反应搜索 (stub)."""
    products = [product] if product else ["P-alldehyde"]
    mock_reactions: list[dict[str, Any]] = [
        {
            "reaction_id": "RXN-00001",
            "scheme": f"{reactant} -> {products[0]} via oxidation",
            "reactants": [reactant],
            "products": products,
            "conditions": "PCC, DCM, rt, 3h",
            "yield_percent": 78.5,
            "reference": "J. Org. Chem. 2018, 83, 12, 6543",
        },
        {
            "reaction_id": "RXN-00002",
            "scheme": f"{reactant} -> {products[0]} via catalytic hydrogenation",
            "reactants": [reactant],
            "products": products,
            "conditions": "Pd/C, H2 (1 atm), EtOH, rt",
            "yield_percent": 92.0,
            "reference": "Org. Lett. 2019, 21, 5, 1234",
        },
    ]
    if reaction_type == "oxidation":
        mock_reactions = [mock_reactions[0]]
    elif reaction_type == "reduction":
        mock_reactions = [mock_reactions[1]]
    return {
        "reactions": mock_reactions[:max_results],
        "total_found": len(mock_reactions),
        "source": "SciFinder (scifinder.cas.org)",
    }


def _reaxys_query_property_registration() -> ToolRegistration:
    """reaxys_query_property — Reaxys 物性数据查询.

    endpoint: reaxys.com, auth: subscription, rate_limit: 20
    """
    return ToolRegistration(
        name="reaxys_query_property",
        description="Reaxys 物性数据查询：检索化学物质的实验物性数据(熔点、沸点、密度、溶解度、折射率等)及对应文献来源。需订阅认证。",
        input_schema={
            "type": "object",
            "properties": {
                "substance": {
                    "type": "string",
                    "description": "物质名称/CAS/SMILES",
                },
                "property_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["melting_point", "boiling_point", "density", "solubility", "refractive_index"]},
                    "default": ["melting_point", "boiling_point", "density"],
                    "description": "请求的物性类型",
                },
            },
            "required": ["substance"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "substance": {"type": "string"},
                "properties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "property_type": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "conditions": {"type": "string"},
                            "reference": {"type": "string"},
                        },
                    },
                    "description": "物性数据列表(含文献来源)",
                },
                "source": {"type": "string"},
            },
            "required": ["substance", "properties", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["reaxys", "property", "chemistry", "experimental", "L3", "connector", "industry_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=900,
            domain_scope=["DOM-A"],
            rate_limit=20,
        ),
    )


async def _reaxys_query_property_handler(
    substance: str,
    property_types: list[str] | None = None,
) -> dict[str, Any]:
    """Reaxys 物性查询 (stub)."""
    if property_types is None:
        property_types = ["melting_point", "boiling_point", "density"]
    mock_props: list[dict[str, Any]] = []
    db: dict[str, dict[str, str]] = {
        "melting_point": {"value": "-114.1", "unit": "°C", "conditions": "1 atm", "reference": "Reaxys ref 12345"},
        "boiling_point": {"value": "78.3", "unit": "°C", "conditions": "1 atm", "reference": "Reaxys ref 12346"},
        "density": {"value": "0.789", "unit": "g/cm³", "conditions": "20 °C", "reference": "Reaxys ref 12347"},
        "solubility": {"value": "miscible", "unit": "", "conditions": "25 °C", "reference": "Reaxys ref 12348"},
        "refractive_index": {"value": "1.3611", "unit": "", "conditions": "20 °C", "reference": "Reaxys ref 12349"},
    }
    for ptype in property_types:
        d = db.get(ptype, {"value": "N/A", "unit": "", "conditions": "", "reference": ""})
        mock_props.append({
            "property_type": ptype,
            "value": d["value"],
            "unit": d["unit"],
            "conditions": d["conditions"],
            "reference": d["reference"],
        })
    return {
        "substance": substance,
        "properties": mock_props,
        "source": "Reaxys (reaxys.com)",
    }


def _thermocalc_phase_diagram_registration() -> ToolRegistration:
    """thermocalc_phase_diagram — Thermo-Calc 相图计算.

    endpoint: thermocalc.com, auth: license, rate_limit: 10, requires_compute: True
    """
    return ToolRegistration(
        name="thermocalc_phase_diagram",
        description="Thermo-Calc 相图计算：基于 CALPHAD 热力学数据库计算多元体系相图，返回相区、相边界和不变反应点。需许可证认证，消耗额外算力。",
        input_schema={
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "体系元素列表，如 ['Fe', 'C']",
                },
                "temperature_range": {
                    "type": "object",
                    "properties": {
                        "t_min": {"type": "number", "minimum": 0},
                        "t_max": {"type": "number", "minimum": 0},
                    },
                    "description": "温度范围 (K)",
                },
                "pressure": {"type": "number", "default": 101325, "description": "压力 (Pa)"},
                "calculation_type": {
                    "type": "string",
                    "enum": ["phase_diagram", "property_diagram", "solidification"],
                    "default": "phase_diagram",
                },
                "database": {"type": "string", "default": "TCFE", "description": "热力学数据库"},
            },
            "required": ["elements"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "system": {"type": "string", "description": "体系标识"},
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "stability_range": {
                                "type": "object",
                                "properties": {
                                    "t_min": {"type": "number"},
                                    "t_max": {"type": "number"},
                                },
                            },
                        },
                    },
                    "description": "稳定相及其温度范围",
                },
                "invariant_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "reaction": {"type": "string"},
                            "composition": {"type": "object"},
                        },
                    },
                    "description": "不变反应点",
                },
                "database": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["system", "phases", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["thermocalc", "phase_diagram", "calphad", "thermodynamics", "L3", "connector", "compute"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=8000,
            domain_scope=["DOM-B"],
            requires_compute=True,
            rate_limit=10,
        ),
    )


async def _thermocalc_phase_diagram_handler(
    elements: list[str],
    temperature_range: dict | None = None,
    pressure: float = 101325,
    calculation_type: str = "phase_diagram",
    database: str = "TCFE",
) -> dict[str, Any]:
    """Thermo-Calc 相图计算 (stub)."""
    if temperature_range is None:
        temperature_range = {"t_min": 300, "t_max": 2000}
    mock_phases: list[dict[str, Any]] = [
        {"name": "BCC_A2", "stability_range": {"t_min": 1185, "t_max": 2000}},
        {"name": "FCC_A1", "stability_range": {"t_min": 912, "t_max": 1394}},
        {"name": "LIQUID", "stability_range": {"t_min": 1811, "t_max": 3000}},
    ]
    mock_invariants: list[dict[str, Any]] = [
        {"temperature": 1811, "reaction": "L <-> BCC_A2 + FCC_A1", "composition": {"C": 0.09}},
        {"temperature": 1394, "reaction": "FCC_A1 <-> BCC_A2", "composition": {"C": 0.0017}},
    ]
    return {
        "system": "-".join(elements),
        "phases": mock_phases,
        "invariant_points": mock_invariants,
        "database": database,
        "source": "Thermo-Calc (thermocalc.com)",
    }


def _vasp_query_result_registration() -> ToolRegistration:
    """vasp_query_result — VASP 计算结果查询.

    endpoint: local HPC, auth: SSH key, rate_limit: 10, requires_compute: True
    """
    return ToolRegistration(
        name="vasp_query_result",
        description="VASP 计算结果查询：从本地 HPC 集群查询 VASP 第一性原理计算任务的能量、力、结构、收敛状态等结果。需 SSH key 认证，消耗额外算力。",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "任务 ID，如 'slurm-12345'",
                },
                "project": {
                    "type": "string",
                    "description": "项目名/工作目录（与 job_id 二选一）",
                },
                "properties": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["energy", "forces", "structure", "convergence", "dos", "band"]},
                    "default": ["energy", "convergence"],
                    "description": "请求的结果属性",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {"type": "string", "enum": ["completed", "running", "failed", "queued"]},
                "energy": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "number", "description": "总能量 (eV)"},
                        "per_atom": {"type": "number"},
                        "encut": {"type": "number", "description": "截断能 (eV)"},
                    },
                },
                "convergence": {
                    "type": "object",
                    "properties": {
                        "electronic": {"type": "boolean"},
                        "ionic": {"type": "boolean"},
                        "iterations": {"type": "integer"},
                    },
                },
                "structure": {
                    "type": "object",
                    "properties": {
                        "formula": {"type": "string"},
                        "lattice": {"type": "string"},
                    },
                },
                "source": {"type": "string"},
            },
            "required": ["job_id", "status", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["vasp", "DFT", "ab_initio", "HPC", "L3", "connector", "compute"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER2,
            estimated_latency_ms=5000,
            domain_scope=["DOM-B", "DOM-C"],
            requires_compute=True,
            rate_limit=10,
        ),
    )


async def _vasp_query_result_handler(
    job_id: str = "",
    project: str = "",
    properties: list[str] | None = None,
) -> dict[str, Any]:
    """VASP 结果查询 (stub)."""
    if properties is None:
        properties = ["energy", "convergence"]
    if not job_id:
        job_id = f"slurm-{project or 'unknown'}"
    result: dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "source": "Local HPC (SSH)",
    }
    if "energy" in properties:
        result["energy"] = {"total": -42.178, "per_atom": -5.272, "encut": 520.0}
    if "convergence" in properties:
        result["convergence"] = {"electronic": True, "ionic": True, "iterations": 24}
    if "structure" in properties:
        result["structure"] = {"formula": "Si2", "lattice": "FCC (diamond cubic)"}
    return result


# ============================================================
# Tier-3 校园数据连接器 (4)
# ============================================================

def _library_search_book_registration() -> ToolRegistration:
    """library_search_book — 图书馆图书搜索.

    endpoint: library.edu/api, auth: campus SSO, rate_limit: 100
    """
    return ToolRegistration(
        name="library_search_book",
        description="图书馆图书搜索：检索校园图书馆馆藏图书，返回书名、作者、ISBN、索书号、馆藏位置和借阅状态。需校园 SSO 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词"},
                "field": {
                    "type": "string",
                    "enum": ["title", "author", "keyword", "isbn"],
                    "default": "keyword",
                    "description": "检索字段",
                },
                "page": {"type": "integer", "minimum": 1, "default": 1},
                "per_page": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "books": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "author": {"type": "string"},
                            "isbn": {"type": "string"},
                            "call_number": {"type": "string", "description": "索书号"},
                            "location": {"type": "string", "description": "馆藏位置"},
                            "available": {"type": "boolean"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "page": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["query", "books", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["library", "book", "campus", "L3", "connector", "campus_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER3,
            estimated_latency_ms=400,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
        ),
    )


async def _library_search_book_handler(
    query: str,
    field: str = "keyword",
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """图书馆图书搜索 (stub)."""
    mock_books: list[dict[str, Any]] = [
        {
            "title": f"Introduction to {query}",
            "author": "Zhang San",
            "isbn": "978-7-0000-0001-2",
            "call_number": "TP311/Z123",
            "location": "Main Library - 3F - Stack A12",
            "available": True,
        },
        {
            "title": f"Advanced {query} textbook",
            "author": "Li Si",
            "isbn": "978-7-0000-0002-9",
            "call_number": "O6/L456",
            "location": "Science Branch Library - 2F",
            "available": False,
        },
    ]
    return {
        "query": query,
        "books": mock_books[:per_page],
        "total_found": len(mock_books),
        "page": page,
        "source": "Campus Library (library.edu/api)",
    }


def _edu_get_curriculum_registration() -> ToolRegistration:
    """edu_get_curriculum — 教务课程信息查询.

    endpoint: jwxt.edu.cn, auth: campus SSO, rate_limit: 50
    """
    return ToolRegistration(
        name="edu_get_curriculum",
        description="教务课程信息查询：从教务系统查询课程安排，包括课程编号、名称、学分、授课教师、上课时间和地点。需校园 SSO 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "semester": {
                    "type": "string",
                    "description": "学期标识，如 '2024-2025-1'",
                },
                "course_id": {"type": "string", "description": "课程编号"},
                "course_name": {"type": "string", "description": "课程名称(模糊匹配)"},
                "department": {"type": "string", "description": "开课院系"},
            },
            "required": ["semester"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "semester": {"type": "string"},
                "courses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "course_id": {"type": "string"},
                            "name": {"type": "string"},
                            "credits": {"type": "number"},
                            "instructor": {"type": "string"},
                            "schedule": {"type": "string", "description": "上课时间"},
                            "location": {"type": "string", "description": "上课地点"},
                            "capacity": {"type": "integer"},
                            "enrolled": {"type": "integer"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["semester", "courses", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["curriculum", "course", "academic_affairs", "campus", "L3", "connector", "campus_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER3,
            estimated_latency_ms=500,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=50,
        ),
    )


async def _edu_get_curriculum_handler(
    semester: str,
    course_id: str = "",
    course_name: str = "",
    department: str = "",
) -> dict[str, Any]:
    """教务课程信息查询 (stub)."""
    mock_courses: list[dict[str, Any]] = [
        {
            "course_id": "CHEM301",
            "name": course_name or "Physical Chemistry",
            "credits": 4.0,
            "instructor": "Prof. Wang",
            "schedule": "Mon 8:00-9:40; Wed 10:00-11:40",
            "location": "Teaching Bldg 3 - Room 201",
            "capacity": 60,
            "enrolled": 54,
        },
        {
            "course_id": "MATS205",
            "name": "Materials Science Fundamentals",
            "credits": 3.0,
            "instructor": "Prof. Liu",
            "schedule": "Tue 14:00-15:40; Fri 8:00-9:40",
            "location": "Lab Bldg 1 - Room 105",
            "capacity": 45,
            "enrolled": 45,
        },
    ]
    if course_id:
        mock_courses = [c for c in mock_courses if c["course_id"] == course_id] or mock_courses
    return {
        "semester": semester,
        "courses": mock_courses,
        "total_found": len(mock_courses),
        "source": "Academic Affairs System (jwxt.edu.cn)",
    }


def _lims_query_experiment_registration() -> ToolRegistration:
    """lims_query_experiment — LIMS 实验数据查询.

    endpoint: lims.lab.edu, auth: lab token, rate_limit: 30
    """
    return ToolRegistration(
        name="lims_query_experiment",
        description="实验室信息管理系统(LIMS)实验数据查询：按实验 ID、样品 ID 或日期范围检索实验记录，返回仪器、测试结果、操作员和时间戳。需实验室 token 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string", "description": "实验编号"},
                "sample_id": {"type": "string", "description": "样品编号"},
                "date_range": {
                    "type": "object",
                    "properties": {
                        "begin": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                        "end": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    },
                    "description": "日期范围",
                },
                "data_type": {
                    "type": "string",
                    "enum": ["xrd", "sem", "tem", "raman", "uv_vis", "all"],
                    "default": "all",
                    "description": "实验数据类型",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "experiment_id": {"type": "string"},
                            "sample_id": {"type": "string"},
                            "instrument": {"type": "string"},
                            "data_type": {"type": "string"},
                            "operator": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "results": {"type": "object", "description": "测试结果键值对"},
                            "report_url": {"type": "string"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["experiments", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["lims", "experiment", "lab", "campus", "L3", "connector", "campus_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER3,
            estimated_latency_ms=600,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=30,
        ),
    )


async def _lims_query_experiment_handler(
    experiment_id: str = "",
    sample_id: str = "",
    date_range: dict | None = None,
    data_type: str = "all",
) -> dict[str, Any]:
    """LIMS 实验数据查询 (stub)."""
    exp_id = experiment_id or "EXP-2024-0001"
    sid = sample_id or "SMP-00042"
    mock_experiments: list[dict[str, Any]] = [
        {
            "experiment_id": exp_id,
            "sample_id": sid,
            "instrument": "Bruker D8 Advance (XRD)",
            "data_type": "xrd",
            "operator": "PhD student Chen",
            "timestamp": "2024-09-15T10:30:00Z",
            "results": {
                "2theta_peaks": [20.86, 26.64, 36.54],
                "identified_phase": "α-Quartz",
                "crystallite_size_nm": 42.3,
            },
            "report_url": "https://lims.lab.edu/reports/EXP-2024-0001.pdf",
        },
    ]
    if data_type != "all":
        mock_experiments = [e for e in mock_experiments if e["data_type"] == data_type] or mock_experiments
    return {
        "experiments": mock_experiments,
        "total_found": len(mock_experiments),
        "source": "LIMS (lims.lab.edu)",
    }


def _campus_kb_search_registration() -> ToolRegistration:
    """campus_kb_search — 校园知识库搜索.

    endpoint: kb.campus.edu, auth: campus SSO, rate_limit: 200
    """
    return ToolRegistration(
        name="campus_kb_search",
        description="校园知识库搜索：检索校园内部知识库文章、规定、操作手册、FAQ 等，返回标题、摘要、分类和链接。需校园 SSO 认证。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "category": {
                    "type": "string",
                    "enum": ["regulation", "manual", "faq", "announcement", "all"],
                    "default": "all",
                    "description": "知识分类",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "snippet": {"type": "string", "description": "内容摘要"},
                            "url": {"type": "string"},
                            "category": {"type": "string"},
                            "updated_at": {"type": "string"},
                            "relevance_score": {"type": "number"},
                        },
                    },
                },
                "total_found": {"type": "integer"},
                "source": {"type": "string"},
            },
            "required": ["query", "articles", "total_found", "source"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["knowledge_base", "campus", "faq", "L3", "connector", "campus_data"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.CONNECTOR_TIER3,
            estimated_latency_ms=300,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=200,
        ),
    )


async def _campus_kb_search_handler(
    query: str,
    category: str = "all",
    max_results: int = 10,
) -> dict[str, Any]:
    """校园知识库搜索 (stub)."""
    mock_articles: list[dict[str, Any]] = [
        {
            "title": f"关于 {query} 的操作指引",
            "snippet": f"本文档详细介绍了 {query} 的标准操作流程和注意事项...",
            "url": "https://kb.campus.edu/article/0001",
            "category": "manual",
            "updated_at": "2024-08-20",
            "relevance_score": 0.95,
        },
        {
            "title": f"{query} 常见问题解答 (FAQ)",
            "snippet": f"汇总了与 {query} 相关的高频问题及解答...",
            "url": "https://kb.campus.edu/article/0002",
            "category": "faq",
            "updated_at": "2024-07-15",
            "relevance_score": 0.81,
        },
    ]
    if category != "all":
        mock_articles = [a for a in mock_articles if a["category"] == category] or mock_articles
    return {
        "query": query,
        "articles": mock_articles[:max_results],
        "total_found": len(mock_articles),
        "source": "Campus Knowledge Base (kb.campus.edu)",
    }


# ============================================================
# 工具注册信息列表
# ============================================================

CONNECTOR_TOOL_DEFINITIONS: list[tuple[ToolRegistration, Any]] = [
    # Tier-1 公共数据连接器 (10)
    (_nist_query_spectrum_registration(), _nist_query_spectrum_handler),
    (_mp_query_material_registration(), _mp_query_material_handler),
    (_ss_search_paper_registration(), _ss_search_paper_handler),
    (_cie_get_colorimetry_registration(), _cie_get_colorimetry_handler),
    (_icdd_xrd_match_registration(), _icdd_xrd_match_handler),
    (_pubchem_query_compound_registration(), _pubchem_query_compound_handler),
    (_crossref_resolve_doi_registration(), _crossref_resolve_doi_handler),
    (_arxiv_search_paper_registration(), _arxiv_search_paper_handler),
    (_wiki_get_summary_registration(), _wiki_get_summary_handler),
    (_openalex_search_works_registration(), _openalex_search_works_handler),
    # Tier-2 行业数据连接器 (6)
    (_cas_query_substance_registration(), _cas_query_substance_handler),
    (_wos_search_article_registration(), _wos_search_article_handler),
    (_scifinder_react_search_registration(), _scifinder_react_search_handler),
    (_reaxys_query_property_registration(), _reaxys_query_property_handler),
    (_thermocalc_phase_diagram_registration(), _thermocalc_phase_diagram_handler),
    (_vasp_query_result_registration(), _vasp_query_result_handler),
    # Tier-3 校园数据连接器 (4)
    (_library_search_book_registration(), _library_search_book_handler),
    (_edu_get_curriculum_registration(), _edu_get_curriculum_handler),
    (_lims_query_experiment_registration(), _lims_query_experiment_handler),
    (_campus_kb_search_registration(), _campus_kb_search_handler),
]

# 便捷访问
CONNECTOR_TOOL_NAMES = [reg.name for reg, _ in CONNECTOR_TOOL_DEFINITIONS]

# 按层级分类
TIER1_TOOLS = [
    "nist_query_spectrum",
    "mp_query_material",
    "ss_search_paper",
    "cie_get_colorimetry",
    "icdd_xrd_match",
    "pubchem_query_compound",
    "crossref_resolve_doi",
    "arxiv_search_paper",
    "wiki_get_summary",
    "openalex_search_works",
]

TIER2_TOOLS = [
    "cas_query_substance",
    "wos_search_article",
    "scifinder_react_search",
    "reaxys_query_property",
    "thermocalc_phase_diagram",
    "vasp_query_result",
]

TIER3_TOOLS = [
    "library_search_book",
    "edu_get_curriculum",
    "lims_query_experiment",
    "campus_kb_search",
]


def get_connector_tool(name: str) -> tuple[ToolRegistration, Any] | None:
    """按名称获取连接器工具定义.

    Args:
        name: 工具名称

    Returns:
        (ToolRegistration, handler) 元组，未找到时返回 None
    """
    for reg, handler in CONNECTOR_TOOL_DEFINITIONS:
        if reg.name == name:
            return reg, handler
    return None
