"""L7 渲染器 — TableRenderer (application/vnd.dy3.table+json).

将结构化表格数据 Artifact 渲染为可交互 HTML 表格。
服务端完成数据归一化、条件格式规则与 42 KP 学情微图嵌入。

实现能力 (对应 L7 设计文档 §2.6 + §4.1.3):

1. **表格核心** (§2.6.1): 排序 / 筛选 / 分页 (10/25/50/100) /
   条件格式 / 列固定 / CSV 导出 (前端交互由 config 驱动)。
2. **42 KP × 4 参数微型条形图** (§2.6.2): payload["bkt_table"] 时,
   每行 = 一个 KP, 列 = P(L)/P(K|L)/P(G)/P(S) 四参数,
   单元格以彩色条形图呈现 (表格即图表)。
3. **条件格式规则** (§2.6.1): payload["format_rules"] 定义
   基于值的颜色标注 (如 量子效率 >80% 绿 / <30% 红)。
4. **BKT 学情着色**: KP 行按掌握度着色, 瓶颈 KP 红色脉冲。

融合世界先进方案:
    - 表格可访问性: 语义化 <table> + scope/th + caption
    - 微型条形图: 高信息密度表格设计 (Tufte 数据墨水比)

输出契约:
    RenderDescriptor.html   — 完整可交互表格 HTML
    RenderDescriptor.config — {columns, sortable, pagination, csv_export, mini_bars}
    RenderDescriptor.assets — []
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Artifact, RenderContext
from ._common import (
    BKT_PARAM_KEYS,
    build_descriptor,
    esc,
    get_bkt_state,
    get_kp_state,
    is_bottleneck,
    mastery_color,
    wrap,
)

#: 支持的 MIME 类型
_MIME_TYPES: list[str] = ["application/vnd.dy3.table+json"]

#: 条件格式操作符
_OPS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}

#: BKT 参数显示名
_BKT_LABELS: dict[str, str] = {
    "p_l": "P(L)",
    "p_k_l": "P(K|L)",
    "p_g": "P(G)",
    "p_s": "P(S)",
}


class TableRenderer:
    """表格渲染器 — 结构化数据 → 可交互 HTML 表格."""

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Table artifact requires non-empty payload"
            )
        payload = artifact.payload
        if "headers" not in payload or "rows" not in payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload",
                missing_fields=["headers", "rows"],
                detail="Table artifact requires 'headers' and 'rows' in payload",
            )

        theme = (context.theme if context else "light") or "light"
        headers: list[str] = [str(h) for h in payload.get("headers") or []]
        rows: list[list[Any]] = payload.get("rows") or []
        format_rules = payload.get("format_rules") or []
        bkt_table = payload.get("bkt_table")
        title = str(payload.get("title", "数据表"))
        pagination = payload.get("pagination", {})
        fixed_cols = int(payload.get("fixed_columns", 0) or 0)

        bkt_state = get_bkt_state(artifact, context)
        mini_bars = self._build_mini_bars(bkt_table, bkt_state, theme)

        html_body = self._build_table(
            title, headers, rows, format_rules, bkt_table, mini_bars, fixed_cols, theme
        )
        html = wrap(html_body, "l7-table", theme)

        config = {
            "title": title,
            "columns": headers,
            "sortable": bool(payload.get("sortable", True)),
            "pagination": {
                "enabled": bool(pagination.get("enabled", len(rows) > 25)),
                "page_sizes": pagination.get("page_sizes", [10, 25, 50, 100]),
                "default_size": int(pagination.get("default_size", 25)),
            },
            "csv_export": bool(payload.get("csv_export", True)),
            "filters": bool(payload.get("filters", True)),
            "mini_bars": mini_bars is not None,
            "fixed_columns": fixed_cols,
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=[],
            metadata={
                "renderer": "TableRenderer",
                "row_count": len(rows),
                "col_count": len(headers),
                "mini_bars": mini_bars is not None,
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------

    def _build_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[Any]],
        format_rules: list[dict[str, Any]],
        bkt_table: bool,
        mini_bars: list[dict[str, Any]] | None,
        fixed_cols: int,
        theme: str,
    ) -> str:
        """构建表格 HTML."""
        parts: list[str] = []
        if title:
            parts.append(f'<h3 class="l7-table-title">{esc(title)}</h3>')
        parts.append('<div class="l7-table-scroll">')
        parts.append(
            f'<table class="l7-data-table" data-sortable="true" '
            f'data-page-size="25" data-fixed-cols="{fixed_cols}">'
        )
        parts.append(f"<caption>{esc(title)}</caption>")
        parts.append("<thead><tr>")
        for idx, h in enumerate(headers):
            fixed = ' class="col-fixed"' if idx < fixed_cols else ""
            parts.append(
                f'<th scope="col"{fixed} data-col-index="{idx}" '
                f'data-sort-key="{esc(h)}">{esc(h)}<span class="sort-indicator"></span></th>'
            )
        if mini_bars is not None:
            for key in BKT_PARAM_KEYS:
                parts.append(
                    f'<th scope="col" class="mini-bar-header">{_BKT_LABELS[key]}</th>'
                )
        parts.append("</tr></thead>")
        parts.append("<tbody>")

        for r_idx, row in enumerate(rows):
            cells = list(row)
            kp_id = self._row_kp_id(cells, headers)
            row_mini = self._row_mini(mini_bars or [], kp_id)
            row_style = ""
            if row_mini is not None:
                # 行级掌握度着色 (瓶颈红色脉冲)
                p_l = float(row_mini.get("p_l", 0.0) or 0.0)
                bn = is_bottleneck(
                    {k: float(row_mini.get(k, 0.0) or 0.0) for k in BKT_PARAM_KEYS}
                )
                color = mastery_color(p_l, theme)
                row_style = f' style="--kp-mastery:{color}"'
                if bn:
                    row_style = f' class="bkt-bottleneck-pulse" style="--kp-mastery:{color}"'
            parts.append(f"<tr{row_style}>")
            for c_idx, cell in enumerate(cells):
                fixed = ' class="col-fixed"' if c_idx < fixed_cols else ""
                style = self._format_cell(cell, format_rules, c_idx, headers)
                parts.append(f"<td{fixed}{style}>{esc(cell)}</td>")
            if mini_bars is not None:
                for key in BKT_PARAM_KEYS:
                    parts.append(self._mini_cell(row_mini, key, theme))
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        parts.append(
            '<div class="l7-table-footer"><span class="l7-table-count"></span>'
            '<div class="l7-table-pagination"></div></div>'
        )
        return "\n".join(parts)

    @staticmethod
    def _row_kp_id(cells: list[Any], headers: list[str]) -> str:
        """从行中提取 KP ID (匹配 A-01 样式)."""
        for cell in cells:
            s = str(cell)
            if len(s) == 4 and s[0] in "ABCD" and s[1] == "-" and s[2:].isdigit():
                return s
        return ""

    @staticmethod
    def _format_cell(
        cell: Any, rules: list[dict[str, Any]], col_index: int, headers: list[str]
    ) -> str:
        """按格式规则生成单元格样式."""
        if not rules:
            return ""
        try:
            value = float(cell)
        except (TypeError, ValueError):
            return ""
        for rule in rules:
            col = str(rule.get("column", ""))
            if col and col not in headers and not col.isdigit():
                continue
            if col.isdigit() and int(col) != col_index:
                continue
            if col and col in headers and headers.index(col) != col_index:
                continue
            op = rule.get("op", "gte")
            threshold = float(rule.get("threshold", rule.get("value", 0)))
            if _OPS.get(op, _OPS["gte"])(value, threshold):
                color = str(rule.get("color", "#16a34a"))
                return f' style="color:{color};font-weight:600"'
        return ""

    def _build_mini_bars(
        self, bkt_table: Any, bkt_state: dict[str, Any], theme: str
    ) -> list[dict[str, Any]] | None:
        """构建 42 KP × 4 参数微型条形图数据 (§2.6.2).

        Args:
            bkt_table: payload["bkt_table"] — True 或 KP 列表。
            bkt_state: 合并后的 BKT 状态。
            theme: 主题。

        Returns:
            每行一个 {kp_id, p_l, p_k_l, p_g, p_s} 字典；未启用返回 None。
        """
        if not bkt_table:
            return None
        rows: list[dict[str, Any]] = []
        kp_ids: list[str] = []
        if isinstance(bkt_table, list):
            kp_ids = [str(k) for k in bkt_table]
        elif isinstance(bkt_table, dict):
            kp_ids = [str(k) for k in (bkt_table.get("kp_ids") or [])]
        if not kp_ids:
            kp_ids = list(bkt_state.keys())
        for kp_id in kp_ids:
            state = get_kp_state(bkt_state, kp_id)
            if state is None:
                state = {k: 0.0 for k in BKT_PARAM_KEYS}
            rows.append({"kp_id": kp_id, **state})
        return rows if rows else None

    @staticmethod
    def _row_mini(
        mini_bars: list[dict[str, Any]], kp_id: str
    ) -> dict[str, Any] | None:
        """按 KP ID 查找行微型条形图数据."""
        if not kp_id:
            return None
        for row in mini_bars:
            if row.get("kp_id") == kp_id:
                return row
        return None

    @staticmethod
    def _mini_cell(
        row: dict[str, Any] | None, key: str, theme: str
    ) -> str:
        """生成单参数微型条形图单元格."""
        if row is None:
            return '<td class="mini-bar-cell"></td>'
        value = float(row.get(key, 0.0) or 0.0)
        pct = min(max(value, 0.0), 1.0) * 100
        color = mastery_color(value, theme)
        bar = (
            f'<div class="mini-bar" role="img" aria-label="{_BKT_LABELS[key]}={value:.2f}">'
            f'<div class="mini-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="mini-bar-value">{value:.2f}</span></div>'
        )
        return f'<td class="mini-bar-cell">{bar}</td>'
