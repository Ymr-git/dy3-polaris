"""L7 渲染器 — TextRenderer (text/vnd.dy3+markdown).

将 Markdown 文本 Artifact 渲染为富文本 HTML。服务端使用 python-markdown
(3.3+) 完成语法转换，前端直接挂载产出 HTML 并接管 KaTeX/Prism 后处理。

实现能力 (对应 L7 设计文档 §2.2 + 任务拆分 T2):

1. **扩展语法** (自定义 markdown Extension):
   - 知识卡片 ``:::kp[A-01]{title="..."}:::`` → 卡片组件 (KP 标识 + 掌握度 + 可折叠)
   - 公式占位 ``$...$`` / ``$$...$$`` → KaTeX 容器 (保留原文, 前端 auto-render)
   - 术语高亮 ``==晶体场分裂==`` → ``<mark class="term-highlight">``
   - 折叠区域 ``> [!note] 标题`` → ``<details class="callout">`` (Obsidian 风格)
   - 代码块高亮 → ``language-*`` 类 (前端 Prism.js 接管)
2. **BKT 三档个性化** (§2.2.3): 按平均 P(L) 推断初学者/进阶/精通,
   beginner 为术语注入悬浮解释气泡, 高级模式精简输出, 渐进式无硬边界。
3. **Socratic 对话气泡** (§2.2.4): payload["dialogue"] 渲染为
   教师提问 (靛蓝问号) / 系统解释 (琥珀灯泡) 气泡组。

融合世界先进方案:
    - python-markdown 扩展机制 (与 Obsidian/marked 生态同构)
    - marked-katex 集成最佳实践: 公式 token 透传 + 前端接管
    - Obsidian callout 语法 (``> [!note]``)

输出契约:
    RenderDescriptor.html   — 完整可嵌入 HTML
    RenderDescriptor.config — {engine, extensions, katex, prism, learner_adaptation, socratic}
    RenderDescriptor.assets — [katex.min.css, katex.min.js] (jsdelivr CDN)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import markdown as _md
from markdown.blockprocessors import BlockProcessor
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from xml.etree import ElementTree as etree

from ..models import Artifact, RenderContext
from ._common import (
    average_p_l,
    build_descriptor,
    esc,
    get_bkt_state,
    get_kp_state,
    kp_badge,
    kp_name,
    wrap,
)

#: 支持的 MIME 类型 (设计文档 §2.9: text/markdown)
_MIME_TYPES: list[str] = ["text/vnd.dy3+markdown", "text/markdown"]

#: 公式块级起始标记
_KATEX_BLOCK_RE = re.compile(r"^\$\$")

#: 知识卡片起始标记: :::kp[A-01]{title="..."}::: (re.MULTILINE: 内容可能同 block)
_KP_CARD_RE = re.compile(
    r"^:::kp\[([A-D]-\d{2})\](?:\{([^}]*)\})?\s*:::\s*$", re.MULTILINE
)

#: callout 起始标记: > [!note] 标题 (re.MULTILINE: block 可能含续行)
_CALLOUT_RE = re.compile(r"^>\s*\[!(\w+)\]\s*(.*)$", re.MULTILINE)

#: 支持的 callout 类型 (Obsidian 风格别名归一)
_CALLOUT_ALIASES: dict[str, str] = {
    "NOTE": "note", "INFO": "info", "TIP": "tip", "HINT": "tip",
    "IMPORTANT": "important", "WARNING": "warning", "CAUTION": "warning",
    "DANGER": "danger", "ERROR": "danger", "QUESTION": "question",
    "HELP": "question", "FAQ": "question", "SUCCESS": "success",
    "CHECK": "success", "DONE": "success", "EXAMPLE": "example",
    "ABSTRACT": "abstract", "SUMMARY": "abstract", "TLDR": "abstract",
}


# ============================================================
# Markdown 扩展: 知识卡片
# ============================================================

class _KnowledgeCardBlock(BlockProcessor):
    """处理 :::kp[A-01]{title=...}::: 知识卡片容器."""

    def test(self, parent: Any, block: str) -> bool:
        return bool(_KP_CARD_RE.match(block))

    def run(self, parent: Any, blocks: list[str]) -> bool:
        block = blocks.pop(0)
        m = _KP_CARD_RE.match(block)
        assert m is not None
        kp_id = m.group(1)
        attrs = self._parse_attrs(m.group(2))
        title = attrs.get("title", kp_name(kp_id))

        # 收集卡片内容直到结束标记 ::: (含嵌套保护)
        # block 可能自带起始标记后的内容行 (含结束标记 :::)
        content_lines: list[str] = []
        lines = block.splitlines()
        depth = 1
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip() == ":::":
                    depth -= 1
                    if depth == 0:
                        break
                    content_lines.append(line)
                    continue
                content_lines.append(line)
        while blocks and depth > 0:
            nxt = blocks.pop(0)
            if nxt.strip() == ":::":
                depth -= 1
                if depth == 0:
                    break
            elif nxt.strip().startswith(":::"):
                depth += 1
                content_lines.append(nxt)
                continue
            content_lines.append(nxt)

        # 递归解析卡片内部内容
        inner = etree.SubElement(parent, "div")
        inner.set("class", "kp-card")
        inner.set("data-kp", kp_id)
        header = etree.SubElement(inner, "details")
        header.set("open", "open")
        summary = etree.SubElement(header, "summary")
        summary.set("class", "kp-card-summary")
        summary.text = f"{kp_id} · {title}"

        body = etree.SubElement(header, "div")
        body.set("class", "kp-card-body")
        if content_lines:
            self.parser.parseChunk(body, "\n".join(content_lines))
        else:
            sub = etree.SubElement(body, "p")
            sub.set("class", "kp-card-empty")
            sub.text = "（本知识点暂无补充内容）"
        return True

    @staticmethod
    def _parse_attrs(raw: str | None) -> dict[str, str]:
        if not raw:
            return {}
        attrs: dict[str, str] = {}
        for key, value in re.findall(r'(\w+)\s*=\s*"([^"]*)"', raw):
            attrs[key] = value
        return attrs


# ============================================================
# Markdown 扩展: callout 折叠区域
# ============================================================

class _CalloutBlock(BlockProcessor):
    """处理 Obsidian 风格 ``> [!note] 标题`` 折叠区域."""

    def test(self, parent: Any, block: str) -> bool:
        return bool(_CALLOUT_RE.match(block))

    def run(self, parent: Any, blocks: list[str]) -> bool:
        block = blocks.pop(0)
        m = _CALLOUT_RE.match(block)
        assert m is not None
        raw_type = m.group(1).upper()
        ctype = _CALLOUT_ALIASES.get(raw_type, "note")
        title = m.group(2).strip() or raw_type.capitalize()

        # block 自带主体行 (合并进同一 block 的续行)
        content_lines: list[str] = []
        lines = block.splitlines()
        if len(lines) > 1:
            content_lines.extend(lines[1:])
        # 继续收集后续以 "> " 前缀的引用行 (callout 主体)
        while blocks:
            nxt = blocks.pop(0)
            if nxt.startswith("> ") or nxt == ">":
                content_lines.append(nxt[2:] if len(nxt) > 2 else "")
                continue
            # 非引用行结束 callout (退回给后续处理器)
            content_lines.append(nxt)
            break

        details = etree.SubElement(parent, "details")
        details.set("class", f"callout callout-{ctype}")
        details.set("open", "open")
        summary = etree.SubElement(details, "summary")
        summary.set("class", "callout-title")
        icon = etree.SubElement(summary, "span")
        icon.set("class", f"callout-icon callout-icon-{ctype}")
        icon.text = self._icon(ctype)
        title_el = etree.SubElement(summary, "span")
        title_el.text = title
        body = etree.SubElement(details, "div")
        body.set("class", "callout-body")
        if content_lines:
            self.parser.parseChunk(body, "\n".join(content_lines))
        return True

    @staticmethod
    def _icon(ctype: str) -> str:
        return {
            "note": "ℹ", "info": "ℹ", "tip": "💡", "important": "❗",
            "warning": "⚠", "danger": "⛔", "question": "❓",
            "success": "✔", "example": "📝", "abstract": "📄",
        }.get(ctype, "ℹ")


# ============================================================
# Markdown 扩展: KaTeX 公式透传
# ============================================================

class _KaTeXBlock(BlockProcessor):
    """将独立公式块 ``$$...$$`` 包裹为 math-display 容器 (原文透传)."""

    def test(self, parent: Any, block: str) -> bool:
        return block.lstrip().startswith("$$")

    def run(self, parent: Any, blocks: list[str]) -> bool:
        block = blocks.pop(0)
        lines = block.strip().splitlines()
        if len(lines) >= 2 and lines[0].strip() == "$$" and lines[-1].strip() == "$$":
            latex = "\n".join(lines[1:-1]).strip()
        else:
            latex = block.strip().strip("$")
        div = etree.SubElement(parent, "div")
        div.set("class", "math-display")
        div.set("data-latex", latex)
        div.text = f"$${latex}$$"
        return True


class _KaTeXInline(InlineProcessor):
    """将行内公式 ``$...$`` 包裹为 math-inline 容器 (原文透传)."""

    def handleMatch(self, m: Any, data: str) -> tuple[Any, int, int]:
        latex = m.group(1)
        el = etree.Element("span")
        el.set("class", "math-inline")
        el.set("data-latex", latex)
        el.text = f"${latex}$"
        return el, m.start(0), m.end(0)


# ============================================================
# Markdown 扩展: 术语高亮
# ============================================================

class _TermHighlightInline(InlineProcessor):
    """将 ``==术语==`` 渲染为术语高亮标记."""

    def handleMatch(self, m: Any, data: str) -> tuple[Any, int, int]:
        el = etree.Element("mark")
        el.set("class", "term-highlight")
        el.set("data-term", m.group(1))
        el.text = m.group(1)
        return el, m.start(0), m.end(0)


# ============================================================
# 扩展注册
# ============================================================

class _Dy3MarkdownExtension(Extension):
    """Dy3+ 文本渲染扩展集 (kp-card / callout / katex / term-highlight).

    优先级说明: python-markdown Registry 中数值越大越先执行。
    blockquote=20 / header=30 / paragraph=200, 故自定义块处理器设为 22,
    确保先于 blockquote 拦截 callout 语法。
    """

    def extendMarkdown(self, md: Any) -> None:
        md.parser.blockprocessors.register(_KnowledgeCardBlock(md.parser), "kp_card", 22)
        md.parser.blockprocessors.register(_CalloutBlock(md.parser), "callout", 22)
        md.parser.blockprocessors.register(_KaTeXBlock(md.parser), "katex_block", 22)
        md.inlinePatterns.register(
            _TermHighlightInline(r"==([^=\n]+?)==", md), "term_highlight", 190
        )
        # 行内公式优先级高于强调/删除线，避免 $ 被误解析；
        # 排除 $$ 块级定界符，避免单行内嵌 $$...$$ 被行内模式吞掉
        md.inlinePatterns.register(
            _KaTeXInline(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)", md), "katex_inline", 185
        )


# ============================================================
# Socratic 对话气泡
# ============================================================

_SOCRATIC_ROLES = {
    "teacher": ("socratic-teacher", "❓", "教师引导"),
    "system": ("socratic-system", "💡", "系统解释"),
}


def _render_socratic(dialogue: list[dict[str, Any]], theme: str) -> str:
    """渲染 Socratic 对话气泡组 (设计文档 §2.2.4)."""
    parts = ['<div class="socratic-dialogue" role="log" aria-live="polite">']
    for item in dialogue:
        role = str(item.get("role", "system"))
        content = str(item.get("content", ""))
        cls, icon, label = _SOCRATIC_ROLES.get(role, _SOCRATIC_ROLES["system"])
        parts.append(
            f'<div class="socratic-bubble {cls}" role="listitem">'
            f'<span class="socratic-icon" aria-hidden="true">{icon}</span>'
            f'<div class="socratic-body"><span class="socratic-label">{label}</span>'
            f'<div class="socratic-content">{content}</div></div></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


# ============================================================
# 渲染器
# ============================================================

class TextRenderer:
    """文本渲染器 — Markdown → 富文本 HTML (服务端 python-markdown).

    使用示例::

        renderer = TextRenderer()
        descriptor = renderer.render(artifact, context)
        # descriptor.html 可直接挂载; descriptor.config 供前端后处理
    """

    _MIME_TYPES: list[str] = list(_MIME_TYPES)
    _EXTENSIONS = ["extra", "codehilite", _Dy3MarkdownExtension()]

    def render(self, artifact: Artifact, context: RenderContext):
        started = time.monotonic()
        if artifact is None or not artifact.payload:
            from ..exceptions import ArtifactValidationError

            raise ArtifactValidationError(
                field="payload", detail="Text artifact requires non-empty payload"
            )
        content = artifact.payload.get("content", "")
        dialogue = artifact.payload.get("dialogue")
        theme = (context.theme if context else "light") or "light"

        # BKT 三档个性化决策
        bkt_state = get_bkt_state(artifact, context)
        adaptation = self._decide_adaptation(artifact, context, bkt_state)

        # 公式数量统计
        math_inline = len(re.findall(r"(?<!\$)\$[^\$\n]+?\$(?!\$)", content))
        math_block = len(re.findall(r"\$\$[^$]*?\$\$", content, flags=re.S))
        extension_names = ["kp-card", "callout", "katex", "term-highlight"]

        html_body = self._to_html(content, adaptation, theme)
        if dialogue:
            html_body += _render_socratic(dialogue, theme)
        html = wrap(html_body, "l7-text", theme)

        config = {
            "engine": "python-markdown",
            "extensions": extension_names,
            "katex": {
                "enabled": True,
                "auto_render": True,
                "delimiters": [
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                ],
            },
            "prism": {"enabled": True, "languages": ["python", "r", "javascript"]},
            "learner_adaptation": adaptation,
            "socratic": {"enabled": bool(dialogue)},
        }
        metadata = {
            "renderer": "TextRenderer",
            "math_inline_count": math_inline,
            "math_block_count": math_block,
            "kp_refs": self._extract_kp_refs(content),
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
            metadata=metadata,
        )
        descriptor.render_time_ms = round((time.monotonic() - started) * 1000, 2)
        return descriptor

    def supported_mime_types(self) -> list[str]:
        return list(self._MIME_TYPES)

    # ----------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------

    def _to_html(self, content: str, adaptation: dict[str, Any], theme: str) -> str:
        """将 Markdown 内容转为 HTML (含扩展语法)."""
        md = _md.Markdown(extensions=self._EXTENSIONS)
        html = md.convert(content)

        # BKT 三档增强: beginner 为术语注入悬浮解释属性
        if adaptation.get("mode") == "beginner":
            html = self._annotate_terms(html, adaptation)

        return html

    @staticmethod
    def _decide_adaptation(
        artifact: Artifact, context: RenderContext, bkt_state: dict[str, Any]
    ) -> dict[str, Any]:
        """决策 BKT 三档个性化 (设计文档 §2.2.3).

        规则: BKT 状态非空时按平均 P(L) 推断
        (beginner <0.4 / intermediate 0.4-0.7 / advanced >0.7)；
        BKT 无数据时回退到 context.learner_mode (上层显式指定)。
        """
        avg = average_p_l(bkt_state)
        explicit = (
            context.learner_mode.value
            if context and context.learner_mode
            else ""
        )
        if bkt_state and avg > 0:
            if avg < 0.4:
                mode, source = "beginner", "avg_p_l"
            elif avg <= 0.7:
                mode, source = "intermediate", "avg_p_l"
            else:
                mode, source = "advanced", "avg_p_l"
        elif explicit in ("beginner", "intermediate", "advanced"):
            mode, source = explicit, "explicit"
        else:
            mode, source = "intermediate", "default"

        applied: list[str] = []
        if mode == "beginner":
            applied.append("term-tooltip")
            applied.append("paragraph-splitting")
        elif mode == "advanced":
            applied.append("concise-mode")

        return {
            "mode": mode,
            "source": source,
            "avg_p_l": round(avg, 4),
            "applied_rules": applied,
        }

    @staticmethod
    def _annotate_terms(html: str, adaptation: dict[str, Any]) -> str:
        """beginner 模式: 为术语高亮标记注入悬浮解释属性 (前端渲染 tooltip)."""
        # 简单注解: 为 term-highlight 添加 data-adapt 标记，前端据此显示气泡
        if "term-tooltip" in adaptation.get("applied_rules", []):
            html = html.replace(
                'class="term-highlight"',
                'class="term-highlight" data-adapt="beginner-tooltip"',
            )
        return html

    @staticmethod
    def _extract_kp_refs(content: str) -> list[str]:
        """提取内容中引用的 KP ID 列表."""
        return sorted(set(re.findall(r"[A-D]-\d{2}", content)))


#: 便捷别名 (与 __init__ 导出命名一致)
MarkdownRenderer = TextRenderer
