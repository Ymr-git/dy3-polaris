"""5 个外部工具的 Schema 定义与 Stub 实现.

分类:
- 光谱查询 (1): nist_query_spectrum_ext
- 材料数据 (1): mp_query_material_ext
- 论文搜索 (1): ss_search_paper_ext
- 色度学数据 (1): cie_get_colorimetry_ext
- 衍射匹配 (1): icdd_xrd_match_ext

这些工具是 Tier-1 连接器的详细实现子集，代表完整规范的外部 API 集成，
包含完整的端点、认证和速率限制配置。

每个工具包含完整的 input_schema / output_schema / Dy3ToolAnnotations。
命名规范: 外部工具使用下划线命名并加 _ext 后缀 (与 ToolRegistration pattern 一致)
"""

from __future__ import annotations

import logging
import random
from typing import Any

from ..core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)

logger = logging.getLogger(__name__)


# ============================================================
# 光谱查询工具 (1): NIST Chemistry WebBook
# ============================================================

def _nist_query_spectrum_ext_registration() -> ToolRegistration:
    """external.nist_query_spectrum_ext — NIST Chemistry WebBook 光谱查询.

    通过 NIST Chemistry WebBook API 查询化合物的红外光谱(IR)、质谱(MS)、
    紫外可见光谱(UV-Vis)数据。
    """
    return ToolRegistration(
        name="nist_query_spectrum_ext",
        description=(
            "通过NIST Chemistry WebBook API查询化合物的红外光谱(IR)、质谱(MS)、"
            "紫外可见光谱(UV-Vis)数据。无需认证的公共API。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "compound_name": {
                    "type": "string",
                    "description": "化合物名称，如 'water', 'ethanol', 'benzene'",
                },
                "formula": {
                    "type": "string",
                    "description": "化学分子式，如 'H2O', 'C2H6O' (可选，辅助定位)",
                },
                "cas_number": {
                    "type": "string",
                    "description": "CAS 号，如 '7732-18-5' (可选，辅助定位)",
                },
                "spectrum_type": {
                    "type": "string",
                    "enum": ["ir", "mass", "uv", "all"],
                    "default": "all",
                    "description": "光谱类型：ir=红外, mass=质谱, uv=紫外可见, all=全部",
                },
            },
            "required": ["compound_name"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "spectra": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["ir", "mass", "uv"],
                                "description": "光谱类型",
                            },
                            "peaks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "wavenumber_or_mz": {
                                            "type": "number",
                                            "description": "波数(cm^-1)或质荷比(m/z)",
                                        },
                                        "intensity": {
                                            "type": "number",
                                            "description": "相对强度(0-1)",
                                        },
                                    },
                                    "required": ["wavenumber_or_mz", "intensity"],
                                },
                                "description": "峰列表",
                            },
                            "source": {
                                "type": "string",
                                "description": "数据来源描述",
                            },
                        },
                        "required": ["type", "peaks", "source"],
                    },
                    "description": "查询到的光谱数据列表",
                },
                "compound_info": {
                    "type": "object",
                    "properties": {
                        "formula": {"type": "string", "description": "分子式"},
                        "mw": {"type": "number", "description": "分子量"},
                        "cas": {"type": "string", "description": "CAS 号"},
                        "name": {"type": "string", "description": "标准名称"},
                    },
                    "required": ["formula", "mw", "name"],
                    "description": "化合物基本信息",
                },
                "query_url": {
                    "type": "string",
                    "description": "NIST WebBook 查询 URL",
                },
            },
            "required": ["spectra", "compound_info", "query_url"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["external", "nist", "spectrum", "ir", "mass_spec", "uv_vis", "chemistry", "L3"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.EXTERNAL,
            estimated_latency_ms=800,
            domain_scope=["DOM-A"],
            rate_limit=60,
            requires_compute=False,
        ),
    )


async def _nist_query_spectrum_ext_handler(
    compound_name: str,
    formula: str = "",
    cas_number: str = "",
    spectrum_type: str = "all",
) -> dict[str, Any]:
    """NIST Chemistry WebBook 光谱查询 (stub: 返回模拟光谱数据)."""
    # 模拟化合物信息库
    mock_compounds: dict[str, dict[str, Any]] = {
        "water": {"formula": "H2O", "mw": 18.015, "cas": "7732-18-5", "name": "Water"},
        "ethanol": {"formula": "C2H6O", "mw": 46.069, "cas": "64-17-5", "name": "Ethanol"},
        "benzene": {"formula": "C6H6", "mw": 78.114, "cas": "71-43-2", "name": "Benzene"},
    }

    key = compound_name.lower().strip()
    info = mock_compounds.get(key, {
        "formula": formula or "C?",
        "mw": round(random.uniform(30.0, 200.0), 3),
        "cas": cas_number or f"{random.randint(10, 99)}-{random.randint(10, 99)}-{random.randint(1, 9)}",
        "name": compound_name,
    })

    query_url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={compound_name}&Units=SI"

    types_to_generate = ["ir", "mass", "uv"] if spectrum_type == "all" else [spectrum_type]
    spectra: list[dict[str, Any]] = []

    for stype in types_to_generate:
        if stype == "ir":
            peaks = [
                {"wavenumber_or_mz": round(3300 + random.uniform(-50, 50), 1), "intensity": round(random.uniform(0.7, 1.0), 3)},
                {"wavenumber_or_mz": round(1640 + random.uniform(-20, 20), 1), "intensity": round(random.uniform(0.4, 0.8), 3)},
                {"wavenumber_or_mz": round(1450 + random.uniform(-10, 10), 1), "intensity": round(random.uniform(0.2, 0.5), 3)},
            ]
        elif stype == "mass":
            peaks = [
                {"wavenumber_or_mz": round(info["mw"], 2), "intensity": 1.0},
                {"wavenumber_or_mz": round(info["mw"] - 18, 2), "intensity": round(random.uniform(0.3, 0.7), 3)},
                {"wavenumber_or_mz": round(info["mw"] / 2, 2), "intensity": round(random.uniform(0.1, 0.4), 3)},
            ]
        else:  # uv
            peaks = [
                {"wavenumber_or_mz": round(254 + random.uniform(-5, 5), 1), "intensity": round(random.uniform(0.6, 1.0), 3)},
                {"wavenumber_or_mz": round(200 + random.uniform(-3, 3), 1), "intensity": round(random.uniform(0.5, 0.9), 3)},
            ]
        spectra.append({
            "type": stype,
            "peaks": peaks,
            "source": "NIST Chemistry WebBook",
        })

    return {
        "spectra": spectra,
        "compound_info": info,
        "query_url": query_url,
    }


# ============================================================
# 材料数据查询工具 (1): Materials Project
# ============================================================

def _mp_query_material_ext_registration() -> ToolRegistration:
    """external.mp_query_material_ext — Materials Project 材料数据查询.

    通过 Materials Project API 查询材料的晶体结构、电子结构、热力学性质等数据。
    使用 X-API-KEY 头部认证。
    """
    return ToolRegistration(
        name="mp_query_material_ext",
        description=(
            "通过Materials Project API查询材料的晶体结构、电子结构、热力学性质等数据。"
            "需要API Key认证(X-API-KEY头部)。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "化学分子式，如 'Fe2O3', 'TiO2', 'LiCoO2'",
                },
                "properties": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["structure", "band_gap", "density", "formation_energy_per_atom"],
                    },
                    "default": ["structure"],
                    "description": "请求返回的属性列表",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "返回结果数量上限",
                },
            },
            "required": ["formula"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "materials": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "material_id": {
                                "type": "string",
                                "description": "Materials Project 材料ID，如 'mp-19770'",
                            },
                            "formula_pretty": {
                                "type": "string",
                                "description": "格式化分子式，如 'Fe2O3'",
                            },
                            "structure": {
                                "type": "object",
                                "description": "晶体结构信息",
                                "properties": {
                                    "lattice": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "number"},
                                            "b": {"type": "number"},
                                            "c": {"type": "number"},
                                            "alpha": {"type": "number"},
                                            "beta": {"type": "number"},
                                            "gamma": {"type": "number"},
                                        },
                                    },
                                    "sites": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                },
                            },
                            "band_gap": {
                                "type": "number",
                                "description": "带隙",
                            },
                            "density": {
                                "type": "number",
                                "description": "密度 (g/cm^3)",
                            },
                            "formation_energy_per_atom": {
                                "type": "number",
                                "description": "每原子形成能",
                            },
                            "crystal_system": {
                                "type": "string",
                                "enum": ["cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal", "monoclinic", "triclinic"],
                                "description": "晶系",
                            },
                        },
                        "required": ["material_id", "formula_pretty", "crystal_system"],
                    },
                    "description": "查询到的材料列表",
                },
                "total_count": {
                    "type": "integer",
                    "description": "匹配的材料总数",
                },
            },
            "required": ["materials", "total_count"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["external", "materials_project", "crystal_structure", "band_gap", "thermodynamics", "materials_science", "L3"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.EXTERNAL,
            estimated_latency_ms=1200,
            domain_scope=["DOM-A"],
            rate_limit=100,
            requires_compute=False,
        ),
    )


async def _mp_query_material_ext_handler(
    formula: str,
    properties: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Materials Project 材料数据查询 (stub: 返回模拟材料数据)."""
    if properties is None:
        properties = ["structure"]

    # 模拟材料数据库
    crystal_systems = ["cubic", "tetragonal", "orthorhombic", "hexagonal", "monoclinic"]
    materials: list[dict[str, Any]] = []
    count = min(limit, random.randint(1, 5))

    for i in range(count):
        material: dict[str, Any] = {
            "material_id": f"mp-{random.randint(1000, 99999)}",
            "formula_pretty": formula,
            "crystal_system": random.choice(crystal_systems),
        }

        if "structure" in properties:
            material["structure"] = {
                "lattice": {
                    "a": round(random.uniform(3.0, 10.0), 4),
                    "b": round(random.uniform(3.0, 10.0), 4),
                    "c": round(random.uniform(3.0, 10.0), 4),
                    "alpha": round(random.uniform(60.0, 120.0), 2),
                    "beta": round(random.uniform(60.0, 120.0), 2),
                    "gamma": round(random.uniform(60.0, 120.0), 2),
                },
                "sites": [
                    {"element": "Fe", "xyz": [0.0, 0.0, 0.0]},
                    {"element": "O", "xyz": [0.5, 0.5, 0.5]},
                ],
            }

        if "band_gap" in properties:
            material["band_gap"] = round(random.uniform(0.0, 5.0), 4)

        if "density" in properties:
            material["density"] = round(random.uniform(2.0, 12.0), 4)

        if "formation_energy_per_atom" in properties:
            material["formation_energy_per_atom"] = round(random.uniform(-3.0, 1.0), 4)

        materials.append(material)

    return {
        "materials": materials,
        "total_count": random.randint(count, count + 20),
    }


# ============================================================
# 论文搜索工具 (1): Semantic Scholar
# ============================================================

def _ss_search_paper_ext_registration() -> ToolRegistration:
    """external.ss_search_paper_ext — Semantic Scholar 论文搜索.

    通过 Semantic Scholar Academic Graph API 搜索学术论文，
    获取引用网络和影响力指标。API Key 可选(x-api-key 头部)。
    """
    return ToolRegistration(
        name="ss_search_paper_ext",
        description=(
            "通过Semantic Scholar Academic Graph API搜索学术论文，"
            "获取引用网络和影响力指标。API Key可选(x-api-key头部)。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如 'graph neural network'",
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["title", "abstract", "authors", "year", "citationCount", "influentialCitationCount"],
                    },
                    "default": ["title", "year"],
                    "description": "请求返回的字段列表",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "返回结果数量上限",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "结果偏移量(用于分页)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paperId": {
                                "type": "string",
                                "description": "Semantic Scholar 论文唯一ID",
                            },
                            "title": {
                                "type": "string",
                                "description": "论文标题",
                            },
                            "abstract": {
                                "type": "string",
                                "description": "摘要",
                            },
                            "authors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "作者姓名"},
                                        "authorId": {"type": "string", "description": "作者ID"},
                                    },
                                    "required": ["name"],
                                },
                                "description": "作者列表",
                            },
                            "year": {
                                "type": "integer",
                                "description": "发表年份",
                            },
                            "citationCount": {
                                "type": "integer",
                                "description": "总引用次数",
                            },
                            "influentialCitationCount": {
                                "type": "integer",
                                "description": "影响力引用次数",
                            },
                            "venue": {
                                "type": "string",
                                "description": "发表期刊/会议",
                            },
                        },
                        "required": ["paperId", "title"],
                    },
                    "description": "搜索到的论文列表",
                },
                "total_count": {
                    "type": "integer",
                    "description": "匹配的论文总数",
                },
                "offset": {
                    "type": "integer",
                    "description": "当前偏移量",
                },
            },
            "required": ["papers", "total_count", "offset"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["external", "semantic_scholar", "paper_search", "citation_network", "literature", "academic", "L3"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.EXTERNAL,
            estimated_latency_ms=600,
            domain_scope=["DOM-A", "DOM-B", "DOM-C"],
            rate_limit=100,
            requires_compute=False,
        ),
    )


async def _ss_search_paper_ext_handler(
    query: str,
    fields: list[str] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    """Semantic Scholar 论文搜索 (stub: 返回模拟论文数据)."""
    if fields is None:
        fields = ["title", "year"]

    # 模拟论文数据
    venues = ["Nature", "Science", "JACS", "Physical Review B", "IEEE TPAMI", "NeurIPS", "ICML"]
    author_names = [
        "Zhang Wei", "Li Ming", "John Smith", "Maria Garcia", "Tanaka Hiroshi",
        "Anna Schmidt", "Raj Patel", "Liu Yang", "Kim Soo-jin", "Carlos Mendez",
    ]

    papers: list[dict[str, Any]] = []
    count = min(limit, 5)

    for i in range(count):
        paper: dict[str, Any] = {
            "paperId": f"{random.randint(10, 99)}{''.join(random.choices('abcdef0123456789', k=32))}",
            "title": f"On the {query}: A comprehensive study (Part {offset + i + 1})",
        }

        if "abstract" in fields:
            paper["abstract"] = (
                f"This paper investigates {query} with novel methodology. "
                f"We present theoretical analysis and experimental validation "
                f"demonstrating significant improvements over baseline approaches."
            )

        if "authors" in fields:
            num_authors = random.randint(1, 4)
            paper["authors"] = [
                {"name": author_names[j % len(author_names)], "authorId": str(random.randint(1000000, 9999999))}
                for j in range(num_authors)
            ]

        if "year" in fields:
            paper["year"] = random.randint(2015, 2025)

        if "citationCount" in fields:
            paper["citationCount"] = random.randint(0, 5000)

        if "influentialCitationCount" in fields:
            paper["influentialCitationCount"] = random.randint(0, 200)

        paper["venue"] = random.choice(venues)
        papers.append(paper)

    return {
        "papers": papers,
        "total_count": random.randint(offset + count, offset + count + 500),
        "offset": offset,
    }


# ============================================================
# 色度学数据查询工具 (1): CIE
# ============================================================

def _cie_get_colorimetry_ext_registration() -> ToolRegistration:
    """external.cie_get_colorimetry_ext — CIE 色度学数据查询.

    查询 CIE(国际照明委员会)标准色度学数据，包括标准光源、色匹配函数、
    色度图坐标。无需认证的公共数据。
    """
    return ToolRegistration(
        name="cie_get_colorimetry_ext",
        description=(
            "查询CIE(国际照明委员会)标准色度学数据，包括标准光源、色匹配函数、"
            "色度图坐标。无需认证的公共数据。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["standard_illuminants", "color_matching", "chromaticity"],
                    "description": "数据类型：standard_illuminants=标准光源, color_matching=色匹配函数, chromaticity=色度图坐标",
                },
                "illuminant": {
                    "type": "string",
                    "description": "光源标识，如 'D65', 'A', 'D50' (用于 standard_illuminants 类型)",
                },
                "wavelength_range": {
                    "type": "object",
                    "properties": {
                        "min_nm": {
                            "type": "number",
                            "minimum": 380,
                            "maximum": 780,
                            "default": 380,
                            "description": "最小波长(nm)",
                        },
                        "max_nm": {
                            "type": "number",
                            "minimum": 380,
                            "maximum": 780,
                            "default": 780,
                            "description": "最大波长(nm)",
                        },
                    },
                    "additionalProperties": False,
                    "description": "波长范围，默认 380-780nm",
                },
            },
            "required": ["data_type"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["standard_illuminants", "color_matching", "chromaticity"],
                            "description": "数据类型",
                        },
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "wavelength": {
                                        "type": "number",
                                        "description": "波长(nm)",
                                    },
                                    "x": {
                                        "type": "number",
                                        "description": "CIE XYZ 中的 X 分量(或色度坐标 x)",
                                    },
                                    "y": {
                                        "type": "number",
                                        "description": "CIE XYZ 中的 Y 分量(或色度坐标 y)",
                                    },
                                    "z": {
                                        "type": "number",
                                        "description": "CIE XYZ 中的 Z 分量",
                                    },
                                },
                                "required": ["wavelength", "x", "y", "z"],
                            },
                            "description": "色度学数据值列表",
                        },
                        "illuminant_info": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "光源名称"},
                                "CCT": {"type": "number", "description": "相关色温(CCT, 单位K)"},
                                "chromaticity_coords": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number", "description": "色度坐标 x"},
                                        "y": {"type": "number", "description": "色度坐标 y"},
                                    },
                                    "required": ["x", "y"],
                                },
                            },
                            "required": ["name", "CCT", "chromaticity_coords"],
                            "description": "光源信息",
                        },
                        "source": {
                            "type": "string",
                            "description": "数据来源描述",
                        },
                    },
                    "required": ["type", "values", "source"],
                    "description": "色度学数据",
                },
                "query_metadata": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string"},
                        "illuminant": {"type": "string"},
                        "wavelength_min": {"type": "number"},
                        "wavelength_max": {"type": "number"},
                        "data_points": {"type": "integer"},
                    },
                    "required": ["data_type", "data_points"],
                    "description": "查询元数据",
                },
            },
            "required": ["data", "query_metadata"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["external", "cie", "colorimetry", "color_matching", "illuminant", "optics", "L3"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.EXTERNAL,
            estimated_latency_ms=700,
            domain_scope=["DOM-A"],
            rate_limit=30,
            requires_compute=False,
        ),
    )


async def _cie_get_colorimetry_ext_handler(
    data_type: str,
    illuminant: str = "",
    wavelength_range: dict[str, float] | None = None,
) -> dict[str, Any]:
    """CIE 色度学数据查询 (stub: 返回模拟色度学数据)."""
    if wavelength_range is None:
        wavelength_range = {"min_nm": 380, "max_nm": 780}

    min_nm = wavelength_range.get("min_nm", 380)
    max_nm = wavelength_range.get("max_nm", 780)

    # 模拟标准光源信息
    illuminants: dict[str, dict[str, Any]] = {
        "D65": {"name": "D65", "CCT": 6504, "chromaticity_coords": {"x": 0.3127, "y": 0.3290}},
        "D50": {"name": "D50", "CCT": 5003, "chromaticity_coords": {"x": 0.3457, "y": 0.3585}},
        "A": {"name": "A", "CCT": 2856, "chromaticity_coords": {"x": 0.4476, "y": 0.4074}},
    }
    ill_name = illuminant.upper() if illuminant else "D65"
    ill_info = illuminants.get(ill_name, {
        "name": ill_name,
        "CCT": round(random.uniform(3000, 7000), 0),
        "chromaticity_coords": {"x": round(random.uniform(0.3, 0.45), 4), "y": round(random.uniform(0.3, 0.4), 4)},
    })

    # 生成色匹配函数模拟数据 (CIE 1931 2-degree observer 近似)
    values: list[dict[str, Any]] = []
    step = 5
    wl = min_nm
    while wl <= max_nm:
        if data_type == "color_matching":
            # CIE 1931 色匹配函数的简化模拟
            if 380 <= wl <= 500:
                z = round(1.0 - abs(wl - 450) / 70.0, 4)
                x = round(0.3 * (1.0 - abs(wl - 440) / 60.0), 4)
                y = round(0.05, 4)
            elif 500 <= wl <= 600:
                z = round(0.02, 4)
                x = round(0.3 * (1.0 - abs(wl - 600) / 100.0), 4)
                y = round(1.0 - abs(wl - 555) / 50.0, 4)
            else:
                z = round(0.0, 4)
                x = round(1.0 - abs(wl - 600) / 180.0, 4) if wl <= 600 else round(0.2 * (1.0 - (wl - 600) / 180.0), 4)
                y = round(1.0 - abs(wl - 555) / 100.0, 4)
                y = max(0.0, y)
                x = max(0.0, x)
        elif data_type == "chromaticity":
            # 色度图坐标
            z_raw = round(random.uniform(0.0, 0.2), 4)
            x_raw = round(random.uniform(0.1, 0.7), 4)
            y_raw = round(random.uniform(0.1, 0.7), 4)
            total = x_raw + y_raw + z_raw
            if total > 0:
                x = round(x_raw / total, 4)
                y = round(y_raw / total, 4)
                z = round(z_raw / total, 4)
            else:
                x = y = z = 0.3333
            values.append({"wavelength": wl, "x": x, "y": y, "z": z})
            wl += step
            continue
        else:  # standard_illuminants
            # 标准光源相对功率分布简化模拟
            peak = 550 if ill_name == "D65" else 600
            power = round(max(0.0, 1.0 - abs(wl - peak) / 200.0) + random.uniform(-0.05, 0.05), 4)
            x = round(power, 4)
            y = round(power * 0.95, 4)
            z = round(power * 1.05, 4)

        values.append({"wavelength": wl, "x": max(0.0, x), "y": max(0.0, y), "z": max(0.0, z)})
        wl += step

    data: dict[str, Any] = {
        "type": data_type,
        "values": values,
        "source": "CIE International Commission on Illumination",
    }
    if data_type in ("standard_illuminants", "color_matching"):
        data["illuminant_info"] = ill_info

    return {
        "data": data,
        "query_metadata": {
            "data_type": data_type,
            "illuminant": ill_name,
            "wavelength_min": min_nm,
            "wavelength_max": max_nm,
            "data_points": len(values),
        },
    }


# ============================================================
# 衍射数据匹配工具 (1): ICDD
# ============================================================

def _icdd_xrd_match_ext_registration() -> ToolRegistration:
    """external.icdd_xrd_match_ext — ICDD XRD 衍射数据匹配.

    通过 ICDD(International Centre for Diffraction Data)数据库匹配
    X 射线衍射图谱，鉴定晶体相。使用 Bearer token 订阅认证。
    """
    return ToolRegistration(
        name="icdd_xrd_match_ext",
        description=(
            "通过ICDD(International Centre for Diffraction Data)数据库匹配"
            "X射线衍射图谱，鉴定晶体相。需要订阅Token(Bearer token)认证。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "diffraction_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "two_theta": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 180,
                                "description": "2θ 衍射角(度)",
                            },
                            "intensity": {
                                "type": "number",
                                "minimum": 0,
                                "description": "相对强度",
                            },
                        },
                        "required": ["two_theta", "intensity"],
                    },
                    "minItems": 3,
                    "description": "衍射数据列表(2θ - 强度对)，至少3个数据点",
                },
                "wavelength": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "default": 1.5406,
                    "description": "X射线波长(Å)，默认 Cu Kα1 = 1.5406Å",
                },
                "match_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.85,
                    "description": "匹配阈值(0-1)，仅返回匹配分数 >= 此值的卡片",
                },
            },
            "required": ["diffraction_data"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "card_number": {
                                "type": "string",
                                "description": "ICDD PDF 卡片号，如 '01-089-0691'",
                            },
                            "compound_name": {
                                "type": "string",
                                "description": "化合物名称",
                            },
                            "formula": {
                                "type": "string",
                                "description": "化学分子式",
                            },
                            "space_group": {
                                "type": "string",
                                "description": "空间群，如 'P63/mmc'",
                            },
                            "peak_matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "observed": {
                                            "type": "number",
                                            "description": "观测到的 2θ 值",
                                        },
                                        "reference": {
                                            "type": "number",
                                            "description": "参考卡片 2θ 值",
                                        },
                                        "delta": {
                                            "type": "number",
                                            "description": "2θ 偏差(度)",
                                        },
                                    },
                                    "required": ["observed", "reference", "delta"],
                                },
                                "description": "逐峰匹配详情",
                            },
                            "match_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "总体匹配分数(0-1)",
                            },
                            "reference_code": {
                                "type": "string",
                                "description": "参考代码",
                            },
                        },
                        "required": ["card_number", "compound_name", "formula", "peak_matches", "match_score"],
                    },
                    "description": "匹配到的 ICDD 卡片列表",
                },
                "best_match": {
                    "type": "object",
                    "properties": {
                        "card_number": {"type": "string", "description": "最佳匹配卡片号"},
                        "compound_name": {"type": "string", "description": "化合物名称"},
                        "match_score": {"type": "number", "description": "匹配分数"},
                    },
                    "required": ["card_number", "compound_name", "match_score"],
                    "description": "最佳匹配结果",
                },
                "unmatched_peaks": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "未能匹配的观测峰 2θ 值列表",
                },
            },
            "required": ["matches", "best_match", "unmatched_peaks"],
        },
        annotations=Dy3ToolAnnotations(
            tags=["external", "icdd", "xrd", "diffraction", "phase_identification", "crystallography", "L3"],
            layer=LayerTag.L3_DOMAIN_KNOWLEDGE,
            category=ToolCategory.EXTERNAL,
            estimated_latency_ms=1500,
            domain_scope=["DOM-A"],
            rate_limit=20,
            requires_compute=False,
        ),
    )


async def _icdd_xrd_match_ext_handler(
    diffraction_data: list[dict[str, float]],
    wavelength: float = 1.5406,
    match_threshold: float = 0.85,
) -> dict[str, Any]:
    """ICDD XRD 衍射数据匹配 (stub: 返回模拟匹配结果)."""
    # 模拟 ICDD 参考卡片库
    mock_cards: list[dict[str, Any]] = [
        {
            "card_number": "01-089-0691",
            "compound_name": "Hematite",
            "formula": "Fe2O3",
            "space_group": "R-3c",
            "reference_peaks": [24.14, 33.15, 35.61, 40.85, 49.48, 54.09, 57.43, 62.45, 64.02],
            "reference_code": "ICDD-PDF4+",
        },
        {
            "card_number": "01-071-4673",
            "compound_name": "Magnetite",
            "formula": "Fe3O4",
            "space_group": "Fd-3m",
            "reference_peaks": [18.27, 30.09, 35.44, 37.05, 43.06, 53.49, 56.97, 62.57],
            "reference_code": "ICDD-PDF4+",
        },
        {
            "card_number": "01-077-9955",
            "compound_name": "Rutile",
            "formula": "TiO2",
            "space_group": "P42/mnm",
            "reference_peaks": [27.45, 36.09, 39.19, 41.23, 44.05, 54.32, 56.64, 62.74],
            "reference_code": "ICDD-PDF4+",
        },
    ]

    observed_peaks = [(d["two_theta"], d.get("intensity", 1.0)) for d in diffraction_data]
    observed_2theta = [p[0] for p in observed_peaks]

    matches: list[dict[str, Any]] = []
    unmatched: list[float] = list(observed_2theta)

    for card in mock_cards:
        peak_matches: list[dict[str, float]] = []
        matched_observed: set[int] = set()

        for obs_idx, (obs_2theta, _) in enumerate(observed_peaks):
            # 找最近参考峰
            best_ref = None
            best_delta = 999.0
            for ref_2theta in card["reference_peaks"]:
                delta = abs(obs_2theta - ref_2theta)
                if delta < best_delta:
                    best_delta = delta
                    best_ref = ref_2theta

            if best_ref is not None and best_delta < 0.5:  # 0.5度容差
                peak_matches.append({
                    "observed": round(obs_2theta, 4),
                    "reference": round(best_ref, 4),
                    "delta": round(best_delta, 4),
                })
                matched_observed.add(obs_idx)

        if not peak_matches:
            continue

        # 匹配分数 = 匹配峰数 / max(观测峰数, 参考峰数)
        match_score = len(peak_matches) / max(len(observed_peaks), len(card["reference_peaks"]))
        match_score = round(min(1.0, match_score), 4)

        if match_score >= match_threshold * 0.6:  # stub 稍放宽阈值以演示
            match_entry: dict[str, Any] = {
                "card_number": card["card_number"],
                "compound_name": card["compound_name"],
                "formula": card["formula"],
                "space_group": card["space_group"],
                "peak_matches": peak_matches,
                "match_score": match_score,
                "reference_code": card["reference_code"],
            }
            matches.append(match_entry)

            # 从未匹配列表中移除已匹配的峰
            for idx in matched_observed:
                if observed_2theta[idx] in unmatched:
                    unmatched.remove(observed_2theta[idx])

    # 按匹配分数排序
    matches.sort(key=lambda m: m["match_score"], reverse=True)

    # 过滤达到阈值的
    matches = [m for m in matches if m["match_score"] >= match_threshold * 0.5]

    best_match = None
    if matches:
        top = matches[0]
        best_match = {
            "card_number": top["card_number"],
            "compound_name": top["compound_name"],
            "match_score": top["match_score"],
        }
    else:
        best_match = {
            "card_number": "",
            "compound_name": "No match found",
            "match_score": 0.0,
        }

    return {
        "matches": matches,
        "best_match": best_match,
        "unmatched_peaks": [round(p, 4) for p in unmatched],
    }


# ============================================================
# 工具注册信息列表
# ============================================================

EXTERNAL_TOOL_DEFINITIONS: list[tuple[ToolRegistration, Any]] = [
    (_nist_query_spectrum_ext_registration(), _nist_query_spectrum_ext_handler),
    (_mp_query_material_ext_registration(), _mp_query_material_ext_handler),
    (_ss_search_paper_ext_registration(), _ss_search_paper_ext_handler),
    (_cie_get_colorimetry_ext_registration(), _cie_get_colorimetry_ext_handler),
    (_icdd_xrd_match_ext_registration(), _icdd_xrd_match_ext_handler),
]

# 便捷访问
EXTERNAL_TOOL_NAMES = [reg.name for reg, _ in EXTERNAL_TOOL_DEFINITIONS]

# 按子分类
SPECTROSCOPY_TOOLS = ["nist_query_spectrum_ext"]
MATERIALS_TOOLS = ["mp_query_material_ext"]
LITERATURE_TOOLS = ["ss_search_paper_ext"]
COLORIMETRY_TOOLS = ["cie_get_colorimetry_ext"]
DIFFRACTION_TOOLS = ["icdd_xrd_match_ext"]


def get_external_tool(name: str) -> tuple[ToolRegistration, Any] | None:
    """按名称获取外部工具定义."""
    for reg, handler in EXTERNAL_TOOL_DEFINITIONS:
        if reg.name == name:
            return reg, handler
    return None
