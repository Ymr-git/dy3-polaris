"""L7 渲染器包 — 七大 Native Renderer (任务拆分 T2).

统一入口，导出 7 个原生渲染器与一键注册函数:

    TextRenderer          — text/vnd.dy3+markdown       Markdown 富文本
    ChartRenderer         — application/vnd.dy3.chart+json       ECharts 图表
    GraphRenderer         — application/vnd.dy3.graph+json       vis.js 知识图谱
    MoleculeRenderer      — application/vnd.dy3.molecule+json    3Dmol.js 分子结构
    TableRenderer         — application/vnd.dy3.table+json       交互表格
    FormulaRenderer       — application/vnd.dy3.formula+json     KaTeX 公式
    ProvenanceRenderer    — application/vnd.dy3.provenance+json  溯源三模式

使用::

    from dy3_polaris.l7.renderers import register_native_renderers
    from dy3_polaris.l7.registry import get_registry

    registry = get_registry()
    register_native_renderers(registry)   # 一键注册全部 7 个原生渲染器
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .text_renderer import TextRenderer, MarkdownRenderer
from .chart_renderer import ChartRenderer
from .graph_renderer import GraphRenderer
from .molecule_renderer import MoleculeRenderer
from .table_renderer import TableRenderer
from .formula_renderer import FormulaRenderer
from .provenance_renderer import ProvenanceRenderer

if TYPE_CHECKING:
    from ..registry import RendererRegistry

__all__ = [
    "TextRenderer",
    "MarkdownRenderer",
    "ChartRenderer",
    "GraphRenderer",
    "MoleculeRenderer",
    "TableRenderer",
    "FormulaRenderer",
    "ProvenanceRenderer",
    "register_native_renderers",
    "native_renderer_classes",
]

#: 全部原生渲染器类 (供探测/文档使用)
native_renderer_classes: tuple[type, ...] = (
    TextRenderer,
    ChartRenderer,
    GraphRenderer,
    MoleculeRenderer,
    TableRenderer,
    FormulaRenderer,
    ProvenanceRenderer,
)


def register_native_renderers(registry: "RendererRegistry") -> list[str]:
    """将全部 7 个原生渲染器注册进给定注册中心.

    Args:
        registry: RendererRegistry 实例。

    Returns:
        实际注册的 MIME 类型列表。
    """
    registered: list[str] = []
    for cls in native_renderer_classes:
        renderer = cls()
        mimes = renderer.supported_mime_types()
        registry.register(renderer)  # register() 自动映射全部声明的 MIME
        for mime in mimes:
            if mime not in registered:
                registered.append(mime)
    return registered
