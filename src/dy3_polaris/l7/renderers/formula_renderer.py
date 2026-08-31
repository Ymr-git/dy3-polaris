"""L7 渲染器 — FormulaRenderer (application/vnd.dy3.formula+json).

将 LaTeX 公式 Artifact 渲染为 KaTeX 可消费的 HTML 容器 (设计文档 §2.7)。

能力:
    - 内联公式 (``$...$``) → ``<span class="math-inline">``
    - 独立公式块 (``$$...$$``) → 带编号 ``<div class="math-display">``
    - 块级公式交互: 点击放大 + LaTeX 源码复制 (前端事件绑定)
    - 覆盖内容: BKT 更新方程、晶体场分裂 Dq、Judd-Ofelt、荧光寿命拟合

输出契约:
    RenderDescriptor.html   — KaTeX 容器 HTML (原文透传, 前端 auto-render)
    RenderDescriptor.config — {katex, display_numbers}
    RenderDescriptor.assets — [katex.min.css, katex.min.js, auto-render]
"""

from __future__ import annotations

import re
import time

from ..models import Artifact, RenderContext
from ._common import build_descriptor, esc, wrap

#: 支持的 MIME 类型
_MIME_TYPES: list[str] = [
    "application/vnd.dy3.formula+json",
    "application/vnd.dy3.formula+tex",
]

#: 行内公式
_INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")
#: 块级公式 (含 \begin{...}...\end{...} 与 $$...$$ 两种形式)
_BLOCK_RE = re.compile(r"(\$\$[^$]*?\$\$|\\begin\{[^}]*\}.*?\\end\{[^}]*\})", flags=re.S)


class FormulaRenderer:
    """公式渲染器 — LaTeX → KaTeX 容器 (服务端透传, 前端渲染)."""

    _MIME_TYPES: list[str] = list(_MIME_TYPES)

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Formula artifact requires non-empty payload"
            )
        latex = str(artifact.payload.get("latex", "")).strip()
        display_numbers = bool(artifact.payload.get("display_numbers", True))
        numbering = artifact.payload.get("numbering", {})
        theme = (context.theme if context else "light") or "light"

        html_body = self._to_html(latex, display_numbers, numbering)
        html = wrap(html_body, "l7-formula", theme)

        config = {
            "katex": {
                "enabled": True,
                "auto_render": True,
                "throw_on_error": False,
                "display_numbers": display_numbers,
                "interactions": ["click-zoom", "copy-source"],
            },
        }
        descriptor = build_descriptor(
            artifact,
            html=html,
            config=config,
            assets=[
                "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
                "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
                "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js",
            ],
            metadata={
                "renderer": "FormulaRenderer",
                "inline_count": len(_INLINE_RE.findall(latex)),
                "block_count": len(_BLOCK_RE.findall(latex)),
            },
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------

    def _to_html(
        self, latex: str, display_numbers: bool, numbering: dict[str, Any]
    ) -> str:
        """将 LaTeX 字符串转为 KaTeX 容器 HTML.

        策略: 先保护块级公式, 再处理行内公式, 最后拼接。
        公式原文透传 (不做转义破坏), 前端 KaTeX 接管渲染。
        """
        parts: list[str] = []
        pos = 0
        block_index = 0
        for m in _BLOCK_RE.finditer(latex):
            if m.start() > pos:
                parts.append(self._inline_convert(latex[pos : m.start()]))
            block_latex = m.group(1)
            block_index += 1
            parts.append(self._block_html(block_latex, block_index, display_numbers, numbering))
            pos = m.end()
        if pos < len(latex):
            parts.append(self._inline_convert(latex[pos:]))
        return "\n".join(parts) if parts else self._inline_convert(latex)

    @staticmethod
    def _inline_convert(text: str) -> str:
        """转换行内公式 $...$ → math-inline 容器 (其余文本原样保留)."""
        if not text.strip():
            return ""

        def _rep(m: re.Match[str]) -> str:
            latex = m.group(1)
            return (
                f'<span class="math-inline" data-latex="{esc(latex)}">'
                f"${esc(latex)}$</span>"
            )

        return _INLINE_RE.sub(_rep, esc(text))

    @staticmethod
    def _block_html(
        latex: str,
        index: int,
        display_numbers: bool,
        numbering: dict[str, Any],
    ) -> str:
        """构建独立公式块 HTML (带编号与交互属性)."""
        num = ""
        if display_numbers:
            tag = str(numbering.get(str(index), index))
            num = f'<span class="formula-number">({tag})</span>'
        return (
            f'<div class="math-display formula-block" data-formula-index="{index}" '
            f'data-latex="{esc(latex.strip("$"))}" tabindex="0" role="button" '
            f'aria-label="公式 {index}, 点击放大, Enter 复制源码">'
            f'<span class="math-display-inner">{esc(latex)}</span>{num}</div>'
        )
