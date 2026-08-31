"""外部知识源适配器 — 动态知识识别 (借鉴 Haystack Agentic RAG WebSearch Fallback).

本地知识库无匹配时, 条件路由到外部知识源 (在线检索 API / 内部检索服务),
实现"动态识别": 系统不再局限于预置的几篇文档, 可按配置接入多源。

可插拔: 实现 :class:`KnowledgeSource` 接口即可接入新源。
- HttpSearchSource: 通用 HTTP 搜索源 (POST JSON 到配置 URL, 返回 {results:[{title,content,url}]})
- 未配置任何源时 enabled()=False, 生成 Agent 保持诚实拒绝 (不编造)。

成熟方案对照 (检索增强生成 RAG 生态):
- Haystack ConditionalRouter + SerperDevWebSearch: 本地文档无答案 → 联网检索兜底
- LangChain DocumentLoaders: 多格式文档动态导入知识库 (本项目 /l3/ingest 已提供)
本模块实现其中的"外部检索兜底"一环。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("dy3_polaris.l3.external_kb")


class KnowledgeSource:
    """外部知识源接口."""

    name: str = "base"

    def enabled(self) -> bool:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """检索外部知识, 返回 [{title, content, url}, ...]."""
        raise NotImplementedError


class HttpSearchSource(KnowledgeSource):
    """通用 HTTP 搜索源: POST JSON 到配置端点.

    请求体: {"query": str, "top_k": int, "api_key": str}
    响应体: {"results": [{"title": str, "content": str, "url": str}]}

    通过环境变量配置:
        DY3_EXT_KB_URL   : 搜索端点 URL
        DY3_EXT_KB_KEY   : 搜索服务 API Key
        DY3_EXT_KB_NAME  : 源名称 (默认 "在线检索")
    """

    name = "http"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        name: str = "在线检索",
        timeout: float = 6.0,
    ) -> None:
        import os

        self._url = url or os.environ.get("DY3_EXT_KB_URL", "")
        self._api_key = api_key or os.environ.get("DY3_EXT_KB_KEY", "")
        self._name = name or os.environ.get("DY3_EXT_KB_NAME", "在线检索")
        self._timeout = timeout

    def enabled(self) -> bool:
        return bool(self._url)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            import requests

            resp = requests.post(
                self._url,
                json={"query": str(query), "top_k": top_k, "api_key": self._api_key},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning("外部知识源状态码 %s", resp.status_code)
                return []
            data = resp.json()
            results = data.get("results") if isinstance(data, dict) else data
            if not isinstance(results, list):
                return []
            cleaned: list[dict[str, Any]] = []
            for item in results[:top_k]:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or item.get("snippet") or "")
                if not content:
                    continue
                cleaned.append({
                    "title": str(item.get("title") or "")[:200],
                    "content": content[:1200],
                    "url": str(item.get("url") or ""),
                    "source": self._name,
                })
            return cleaned
        except Exception as exc:  # noqa: BLE001 - 外部源异常不阻断主流程
            logger.warning("外部知识源检索失败: %s", exc)
            return []


def build_external_knowledge_source() -> KnowledgeSource | None:
    """按配置构建外部知识源 (未配置返回 None, 保持本地诚实拒绝)."""
    import os

    if os.environ.get("DY3_EXT_KB_URL"):
        try:
            return HttpSearchSource()
        except Exception as exc:  # noqa: BLE001
            logger.warning("外部知识源构建失败: %s", exc)
    return None
