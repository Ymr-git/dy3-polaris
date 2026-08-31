"""L7 体验呈现层 — REST API 路由层.

基于 Starlette 构建, 将 L7 渲染流水线、Artifact 管理器、渲染器注册中心
等核心功能暴露为 RESTful JSON API。

遵循与 L2/L3/L4/L5/L6 API 一致的设计模式:
- 统一响应格式: ``{"code": 0, "data": ..., "message": ""}``
- CORS 中间件支持
- 异常统一处理 (L7Error → HTTP 状态码映射)
- 资源导向 URL 设计 (RESTful 语义)

融合世界先进方案的 API 设计:
- Jupyter nbformat: MIME Bundle 多模态渲染路由
- VS Code CustomEditor: 渲染实例生命周期 (render/update/destroy)
- React Server Components: 增量更新 (PUT /render/{id})
- Git DAG: 版本树管理 (GET /artifacts/{id}/versions)
- IntersectionObserver: 视口懒加载 (query 参数构造 RenderContext)

端点列表:
- GET  /health:                              L7 健康检查
- POST /render:                              渲染 Artifact
- GET  /render/{artifact_id}:                获取缓存的渲染描述
- PUT  /render/{artifact_id}:                增量更新渲染
- DELETE /render/{artifact_id}:              销毁渲染实例
- GET  /artifacts:                           Artifact 列表 (支持 query 过滤)
- GET  /artifacts/{artifact_id}:             Artifact 详情
- GET  /artifacts/{artifact_id}/versions:    版本历史
- POST /artifacts/{artifact_id}/edit:        提交编辑 (RFC 6902 JSON Patch)
- GET  /registry/mime-types:                 已注册 MIME 类型列表
- GET  /registry/renderers:                  已注册渲染器列表
- GET  /registry/supports/{mime_type}:       检查 MIME 支持
- GET  /stats:                               统计信息

使用示例::

    from dy3_polaris.l7.api import L7Router
    from dy3_polaris.l7 import ArtifactManager, RendererRegistry, IRenderer

    registry = RendererRegistry()
    registry.register(MyRenderer())
    router = L7Router(
        artifact_manager=ArtifactManager(),
        registry=registry,
    )
    app = router.create_app()

    # 独立运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)

    # 或嵌入到主应用
    from starlette.routing import Mount
    main_routes = [Mount("/l7", app=router.create_app())]

注意:
    RenderPipeline (``dy3_polaris.l7.pipeline``) 由另一代理并行实现,
    本模块通过 ``try/except`` 优雅处理其尚不存在的情况 —— 此时回退到
    内置的 ``_FallbackPipeline`` (基于 RendererRegistry + 内存缓存)。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..artifact_manager import ArtifactManager
from ..exceptions import (
    ArtifactNotFoundError,
    ArtifactNotEditableError,
    ArtifactValidationError,
    L7Error,
    RenderContextError,
    RenderTimeoutError,
    RendererNotFoundError,
    UnsupportedMimeError,
    VersionConflictError,
)
from ..models import (
    Artifact,
    ArtifactDiff,
    ArtifactType,
    LearnerMode,
    RenderContext,
    RenderDescriptor,
    Viewport,
)
from ..registry import RendererRegistry, get_registry
from .openapi import openapi_handler, swagger_ui_handler

_logger = logging.getLogger("dy3_polaris.l7.api.router")


# ============================================================
# RenderPipeline 优雅导入 (可能尚未实现)
# ============================================================

try:  # pragma: no cover - 取决于并行实现是否存在
    from ..pipeline import RenderPipeline as _RealRenderPipeline
except ImportError:  # pragma: no cover
    _RealRenderPipeline = None  # type: ignore[assignment,misc]


# ============================================================
# 统一响应
# ============================================================


# 响应信封单点 (SSOT: shared/contract.py)
from dy3_polaris.shared.contract import err as _err, ok as _ok


def _safe_dump(obj: Any) -> Any:
    """安全地将 dataclass / Pydantic 模型 / dict / list 转为可 JSON 序列化的值."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return _safe_dump(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return _safe_dump(obj.model_dump(mode="json"))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_dump(v) for k, v in obj.items()}
    # 枚举
    if hasattr(obj, "value") and isinstance(obj, type) is False:
        value = getattr(obj, "value", None)
        if isinstance(value, str):
            return value
    return str(obj)


# ============================================================
# L7Error → HTTP 状态码映射
# ============================================================


def _status_for_l7_error(exc: L7Error) -> int:
    """将 L7 异常映射到 HTTP 状态码.

    - RendererNotFoundError / UnsupportedMimeError / ArtifactValidationError
      / RenderContextError → 422 (Unprocessable Entity)
    - ArtifactNotFoundError → 404 (Not Found)
    - ArtifactNotEditableError / VersionConflictError → 409 (Conflict)
    - RenderTimeoutError → 504 (Gateway Timeout)
    - 其他 L7Error → 400 (Bad Request)
    """
    if isinstance(
        exc,
        (RendererNotFoundError, UnsupportedMimeError, ArtifactValidationError, RenderContextError),
    ):
        return 422
    if isinstance(exc, ArtifactNotFoundError):
        return 404
    if isinstance(exc, (ArtifactNotEditableError, VersionConflictError)):
        return 409
    if isinstance(exc, RenderTimeoutError):
        return 504
    return 400


def _jsonrpc_code_for(exc: L7Error) -> int:
    """提取 L7 异常的 JSON-RPC 错误码 (兜底 -32500)."""
    method = getattr(exc, "_jsonrpc_code", None)
    if callable(method):
        try:
            return int(method())
        except Exception:  # pragma: no cover
            return -32500
    return -32500


# ============================================================
# 回退渲染流水线 (RenderPipeline 尚未实现时使用)
# ============================================================


class _FallbackPipeline:
    """内置回退渲染流水线 — 基于 RendererRegistry + 内存缓存.

    当 ``dy3_polaris.l7.pipeline.RenderPipeline`` 尚未实现时使用。
    实现 RenderPipeline 的核心子集接口:
    - render(artifact, context) -> RenderDescriptor
    - update(artifact_id, diff) -> RenderDescriptor
    - destroy(artifact_id) -> None
    - get_cached(artifact_id) -> RenderDescriptor | None
    - clear_cache() -> None
    - get_stats() -> dict

    借鉴方案:
    - VS Code CustomEditor: 渲染实例生命周期 (render/update/destroy)
    - React Server Components: 增量更新 + 资源管理
    - IndexedDB L1: 内存缓存层
    """

    def __init__(self, registry: RendererRegistry) -> None:
        self._registry = registry
        # artifact_id -> RenderDescriptor (缓存)
        self._cache: dict[str, RenderDescriptor] = {}
        # artifact_id -> IRenderer (记录渲染时所用渲染器, 供 update 复用)
        self._renderers: dict[str, Any] = {}
        self._render_count = 0
        self._update_count = 0
        self._destroy_count = 0

    def render(
        self,
        artifact: Artifact,
        context: RenderContext,
        *args: Any,
        **kwargs: Any,
    ) -> RenderDescriptor:
        """渲染 Artifact 并缓存结果.

        通过 registry 按 MIME 路由到渲染器, 调用 ``render()`` 生成
        RenderDescriptor, 并以 artifact_id 为键缓存。

        Args:
            artifact: 待渲染的制品。
            context: 渲染上下文。

        Returns:
            前端可消费的 RenderDescriptor。

        Raises:
            RendererNotFoundError: 没有渲染器处理 artifact.mime。
        """
        renderer = self._registry.get_renderer(artifact.mime)
        descriptor = renderer.render(artifact, context)
        self._cache[artifact.artifact_id] = descriptor
        self._renderers[artifact.artifact_id] = renderer
        self._render_count += 1
        return descriptor

    def update(
        self,
        artifact_id: str,
        diff: ArtifactDiff,
        *args: Any,
        **kwargs: Any,
    ) -> RenderDescriptor:
        """基于增量差异更新已有渲染.

        Args:
            artifact_id: Artifact ID。
            diff: Artifact 增量差异。

        Returns:
            更新后的 RenderDescriptor。

        Raises:
            ArtifactNotFoundError: 该 artifact_id 尚未渲染过 (无缓存)。
        """
        renderer = self._renderers.get(artifact_id)
        if renderer is None:
            raise ArtifactNotFoundError(
                artifact_id,
                detail=f"Render instance not found for artifact: {artifact_id}",
            )
        descriptor = renderer.update(diff)
        self._cache[artifact_id] = descriptor
        self._update_count += 1
        return descriptor

    def destroy(self, artifact_id: str, *args: Any, **kwargs: Any) -> None:
        """销毁渲染实例, 释放资源 (幂等).

        对未知 ID 或已销毁的实例不抛异常, 与真实 RenderPipeline 保持一致。

        Args:
            artifact_id: Artifact ID。
        """
        renderer = self._renderers.pop(artifact_id, None)
        if renderer is not None and hasattr(renderer, "destroy"):
            try:
                renderer.destroy()
            except Exception:  # pragma: no cover
                _logger.debug("renderer.destroy() raised", exc_info=True)
        if artifact_id in self._cache:
            self._cache.pop(artifact_id, None)
            self._destroy_count += 1

    def get_cached(self, artifact_id: str) -> RenderDescriptor | None:
        """获取缓存的渲染描述符."""
        return self._cache.get(artifact_id)

    def clear_cache(self) -> None:
        """清空所有缓存."""
        self._cache.clear()
        self._renderers.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取流水线统计信息."""
        return {
            "cached_count": len(self._cache),
            "render_count": self._render_count,
            "update_count": self._update_count,
            "destroy_count": self._destroy_count,
            "mode": "fallback",
        }


def _resolve_pipeline(
    pipeline: Any,
    registry: RendererRegistry,
    artifact_manager: ArtifactManager | None = None,
) -> Any:
    """解析出可用的渲染流水线实例.

    优先级:
    1. 调用方显式传入的 pipeline;
    2. 真实 RenderPipeline (若已实现) — 通过 registry + artifact_manager 构造;
    3. 内置 ``_FallbackPipeline`` (基于 registry + 内存缓存)。
    """
    if pipeline is not None:
        return pipeline
    if _RealRenderPipeline is not None:
        try:
            return _RealRenderPipeline(
                registry=registry,
                artifact_manager=artifact_manager,
            )
        except TypeError:
            # 兼容仅接受 registry 的旧签名
            try:
                return _RealRenderPipeline(registry)
            except Exception:  # pragma: no cover
                _logger.debug("RenderPipeline 构造失败, 回退到 _FallbackPipeline", exc_info=True)
        except Exception:  # pragma: no cover
            _logger.debug("RenderPipeline 构造失败, 回退到 _FallbackPipeline", exc_info=True)
    return _FallbackPipeline(registry)


# ============================================================
# 渲染上下文构造
# ============================================================


def _build_render_context(query_params: Any) -> RenderContext:
    """从 Starlette query_params 构造 RenderContext (缺省字段使用默认值).

    支持的 query 参数:
    - theme: 主题 (light/dark/auto)
    - learner_mode: 学习者模式 (beginner/intermediate/advanced)
    - viewport_width: 视口宽度 (像素)
    - viewport_height: 视口高度 (像素)
    - locale: 语言区域
    """
    theme = query_params.get("theme", "light")
    learner_mode_str = query_params.get("learner_mode", "intermediate")
    locale = query_params.get("locale", "zh-CN")

    try:
        learner_mode = LearnerMode(learner_mode_str)
    except ValueError:
        learner_mode = LearnerMode.INTERMEDIATE

    viewport = Viewport()
    width_str = query_params.get("viewport_width")
    height_str = query_params.get("viewport_height")
    if width_str is not None:
        try:
            viewport = viewport.model_copy(update={"width": int(width_str)})
        except (ValueError, TypeError):
            pass
    if height_str is not None:
        try:
            viewport = viewport.model_copy(update={"height": int(height_str)})
        except (ValueError, TypeError):
            pass

    return RenderContext(
        viewport=viewport,
        theme=theme,
        learner_mode=learner_mode,
        locale=locale,
    )


# ============================================================
# 路由处理器
# ============================================================


class _RouteHandlers:
    """将 L7 服务方法适配为 Starlette Request→Response 处理器."""

    def __init__(
        self,
        artifact_manager: ArtifactManager,
        registry: RendererRegistry,
        pipeline: Any,
    ) -> None:
        self._am = artifact_manager
        self._registry = registry
        self._pipeline = pipeline

    # ---- 健康检查 ----

    async def health(self, request: Request) -> JSONResponse:
        """GET /health — L7 体验呈现层健康检查."""
        services: dict[str, str] = {
            "registry": "available" if self._registry is not None else "unavailable",
            "artifact_manager": "available" if self._am is not None else "unavailable",
            "pipeline": "available" if self._pipeline is not None else "unavailable",
        }
        pipeline_mode = "fallback"
        if self._pipeline is not None and not isinstance(self._pipeline, _FallbackPipeline):
            pipeline_mode = "real"
        return JSONResponse(_ok({
            "status": "healthy",
            "layer": "L7",
            "timestamp": time.time(),
            "services": services,
            "pipeline_mode": pipeline_mode,
        }))

    # ---- 渲染: POST /render ----

    async def render(self, request: Request) -> JSONResponse:
        """POST /render — 渲染 Artifact.

        请求体: Artifact JSON (字段同 ``Artifact.to_dict()``)
        Query 参数 (可选): theme / learner_mode / viewport_width /
                          viewport_height / locale

        响应: RenderDescriptor
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        if not body:
            return JSONResponse(_err(-32700, "请求体为空"), status_code=400)

        try:
            artifact = Artifact.model_validate(body)
        except Exception as e:
            return JSONResponse(
                _err(-32503, "Artifact 解析失败", str(e)),
                status_code=422,
            )

        context = _build_render_context(request.query_params)

        try:
            # 自动注册 Artifact 到 ArtifactManager (幂等)
            try:
                self._am.register(artifact)
            except Exception:  # pragma: no cover
                _logger.debug("Artifact 自动注册跳过 (可能已注册)")

            descriptor = self._pipeline.render(artifact, context)
            return JSONResponse(_ok(_safe_dump(descriptor)))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("渲染失败")
            return JSONResponse(
                _err(-32400, "渲染失败", str(e)),
                status_code=500,
            )

    # ---- 渲染: GET /render/{artifact_id} ----

    async def get_cached_render(self, request: Request) -> JSONResponse:
        """GET /render/{artifact_id} — 获取缓存的渲染描述."""
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            descriptor = self._pipeline.get_cached(artifact_id)
            if descriptor is None:
                return JSONResponse(
                    _err(-32502, "渲染缓存不存在", f"artifact_id={artifact_id}"),
                    status_code=404,
                )
            return JSONResponse(_ok(_safe_dump(descriptor)))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("获取缓存渲染失败")
            return JSONResponse(
                _err(-32400, "获取缓存渲染失败", str(e)),
                status_code=500,
            )

    # ---- 渲染: PUT /render/{artifact_id} ----

    async def update_render(self, request: Request) -> JSONResponse:
        """PUT /render/{artifact_id} — 增量更新渲染.

        请求体: ArtifactDiff JSON (字段同 ``ArtifactDiff.to_dict()``)
        """
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            diff = ArtifactDiff.model_validate(body)
        except Exception as e:
            return JSONResponse(
                _err(-32503, "ArtifactDiff 解析失败", str(e)),
                status_code=422,
            )

        try:
            descriptor = self._pipeline.update(artifact_id, diff)
            return JSONResponse(_ok(_safe_dump(descriptor)))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("更新渲染失败")
            return JSONResponse(
                _err(-32400, "更新渲染失败", str(e)),
                status_code=500,
            )

    # ---- 渲染: DELETE /render/{artifact_id} ----

    async def destroy_render(self, request: Request) -> JSONResponse:
        """DELETE /render/{artifact_id} — 销毁渲染实例."""
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            self._pipeline.destroy(artifact_id)
            return JSONResponse(_ok({"artifact_id": artifact_id, "destroyed": True}))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("销毁渲染失败")
            return JSONResponse(
                _err(-32400, "销毁渲染失败", str(e)),
                status_code=500,
            )

    # ---- Artifact: GET /artifacts ----

    async def list_artifacts(self, request: Request) -> JSONResponse:
        """GET /artifacts — Artifact 列表 (支持 query 过滤 + 分页排序).

        Query 参数 (可选): type / source_agent / session_id / kp_id
                          page (默认 1) / size (默认 20)
                          sort (如 -created_at / created_at / title)
        """
        artifact_type = request.query_params.get("type")
        source_agent = request.query_params.get("source_agent")
        session_id = request.query_params.get("session_id")
        kp_id = request.query_params.get("kp_id")

        # 将字符串 type 转为 ArtifactType (若可识别)
        type_filter: ArtifactType | str | None = artifact_type
        if artifact_type is not None:
            try:
                type_filter = ArtifactType(artifact_type)
            except ValueError:
                type_filter = artifact_type

        # 分页参数
        try:
            page = max(1, int(request.query_params.get("page", "1")))
        except (ValueError, TypeError):
            page = 1
        try:
            size = max(1, int(request.query_params.get("size", "20")))
        except (ValueError, TypeError):
            size = 20

        # 排序参数
        sort = request.query_params.get("sort", "")
        reverse = sort.startswith("-")
        sort_field = sort.lstrip("-") if sort else ""

        try:
            artifacts = self._am.list_artifacts(
                session_id=session_id,
                artifact_type=type_filter,
                source_agent=source_agent,
                kp_id=kp_id,
            )

            # 排序
            if sort_field:
                artifacts.sort(
                    key=lambda a: getattr(a, sort_field, "") or "",
                    reverse=reverse,
                )

            # 分页
            total = len(artifacts)
            total_pages = (total + size - 1) // size if total > 0 else 0
            start = (page - 1) * size
            end = start + size
            page_items = artifacts[start:end]

            return JSONResponse(_ok({
                "items": [_safe_dump(a) for a in page_items],
                "total": total,
                "page": page,
                "size": size,
                "total_pages": total_pages,
            }))
        except Exception as e:
            _logger.exception("列出 Artifact 失败")
            return JSONResponse(
                _err(-32400, "列出 Artifact 失败", str(e)),
                status_code=500,
            )

    # ---- Artifact: GET /artifacts/{artifact_id} ----

    async def get_artifact(self, request: Request) -> JSONResponse:
        """GET /artifacts/{artifact_id} — Artifact 详情."""
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            artifact = self._am.get(artifact_id)
            return JSONResponse(_ok(_safe_dump(artifact)))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("获取 Artifact 失败")
            return JSONResponse(
                _err(-32400, "获取 Artifact 失败", str(e)),
                status_code=500,
            )

    # ---- Artifact: GET /artifacts/{artifact_id}/versions ----

    async def get_versions(self, request: Request) -> JSONResponse:
        """GET /artifacts/{artifact_id}/versions — 版本历史 (拓扑排序)."""
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            nodes = self._am.get_version_history(artifact_id)
            return JSONResponse(_ok([_safe_dump(n) for n in nodes]))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("获取版本历史失败")
            return JSONResponse(
                _err(-32400, "获取版本历史失败", str(e)),
                status_code=500,
            )

    # ---- Artifact: POST /artifacts/{artifact_id}/edit ----

    async def edit_artifact(self, request: Request) -> JSONResponse:
        """POST /artifacts/{artifact_id}/edit — 提交编辑 (RFC 6902 JSON Patch).

        请求体: ArtifactDiff JSON (字段同 ``ArtifactDiff.to_dict()``)
        """
        artifact_id = request.path_params.get("artifact_id", "")
        if not artifact_id:
            return JSONResponse(_err(-32700, "缺少路径参数: artifact_id"), status_code=400)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_err(-32700, "请求体解析失败"), status_code=400)

        try:
            diff = ArtifactDiff.model_validate(body)
        except Exception as e:
            return JSONResponse(
                _err(-32503, "ArtifactDiff 解析失败", str(e)),
                status_code=422,
            )

        try:
            edited = self._am.apply_edit(artifact_id, diff)
            return JSONResponse(_ok(_safe_dump(edited)))
        except L7Error as e:
            return JSONResponse(
                _err(_jsonrpc_code_for(e), e.detail or e.code, str(e)),
                status_code=_status_for_l7_error(e),
            )
        except Exception as e:
            _logger.exception("应用编辑失败")
            return JSONResponse(
                _err(-32400, "应用编辑失败", str(e)),
                status_code=500,
            )

    # ---- Registry: GET /registry/mime-types ----

    async def list_mime_types(self, request: Request) -> JSONResponse:
        """GET /registry/mime-types — 已注册 MIME 类型列表."""
        try:
            mimes = self._registry.list_mime_types()
            return JSONResponse(_ok(mimes))
        except Exception as e:
            _logger.exception("列出 MIME 类型失败")
            return JSONResponse(
                _err(-32400, "列出 MIME 类型失败", str(e)),
                status_code=500,
            )

    # ---- Registry: GET /registry/renderers ----

    async def list_renderers(self, request: Request) -> JSONResponse:
        """GET /registry/renderers — 已注册渲染器列表 (类名)."""
        try:
            names = self._registry.list_renderers()
            return JSONResponse(_ok(names))
        except Exception as e:
            _logger.exception("列出渲染器失败")
            return JSONResponse(
                _err(-32400, "列出渲染器失败", str(e)),
                status_code=500,
            )

    # ---- Registry: GET /registry/supports/{mime_type} ----

    async def supports_mime(self, request: Request) -> JSONResponse:
        """GET /registry/supports/{mime_type} — 检查 MIME 支持."""
        mime_type = request.path_params.get("mime_type", "")
        if not mime_type:
            return JSONResponse(_err(-32700, "缺少路径参数: mime_type"), status_code=400)

        try:
            supported = self._registry.is_supported(mime_type)
            return JSONResponse(_ok({"mime_type": mime_type, "supported": bool(supported)}))
        except Exception as e:
            _logger.exception("检查 MIME 支持失败")
            return JSONResponse(
                _err(-32400, "检查 MIME 支持失败", str(e)),
                status_code=500,
            )

    # ---- 统计: GET /stats ----

    async def stats(self, request: Request) -> JSONResponse:
        """GET /stats — 统计信息 (pipeline + artifact)."""
        try:
            pipeline_stats: dict[str, Any] = {}
            if self._pipeline is not None and hasattr(self._pipeline, "get_stats"):
                pipeline_stats = self._pipeline.get_stats() or {}

            artifact_stats: dict[str, Any] = {}
            if self._am is not None and hasattr(self._am, "get_stats"):
                artifact_stats = self._am.get_stats() or {}

            return JSONResponse(_ok({
                "pipeline": _safe_dump(pipeline_stats),
                "artifacts": _safe_dump(artifact_stats),
                "registry_size": self._registry.size if self._registry is not None else 0,
            }))
        except Exception as e:
            _logger.exception("获取统计信息失败")
            return JSONResponse(
                _err(-32400, "获取统计信息失败", str(e)),
                status_code=500,
            )


# ============================================================
# L7Router
# ============================================================


class L7Router:
    """L7 体验呈现层 REST API 路由器.

    将 ArtifactManager / RendererRegistry / RenderPipeline 的核心功能
    暴露为 RESTful API。遵循与 L2Router / L3Router / L5Router 一致的设计模式。

    使用示例::

        from dy3_polaris.l7.api import L7Router
        from dy3_polaris.l7 import ArtifactManager, RendererRegistry

        registry = RendererRegistry()
        registry.register(MyRenderer())
        router = L7Router(
            artifact_manager=ArtifactManager(),
            registry=registry,
        )
        app = router.create_app()

        # 独立运行
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8007)

        # 或嵌入到主应用
        from starlette.routing import Mount
        main_routes = [Mount("/l7", app=router.create_app())]

    Args:
        artifact_manager: L7 Artifact 管理器 (可选, 缺省新建).
        registry: L7 渲染器注册中心 (可选, 缺省使用全局单例).
        pipeline: L7 渲染流水线 (可选, 缺省惰性创建; 若
            ``RenderPipeline`` 尚未实现则回退到内置 ``_FallbackPipeline``).
        cors_origins: CORS 允许的源 (默认 ["*"]).
    """

    def __init__(
        self,
        artifact_manager: ArtifactManager | None = None,
        registry: RendererRegistry | None = None,
        pipeline: Any | None = None,
        cors_origins: list[str] | None = None,
        api_prefix: str = "",
    ) -> None:
        """初始化 L7 路由器.

        Args:
            artifact_manager: L7 Artifact 管理器 (可选, 缺省新建).
            registry: L7 渲染器注册中心 (可选, 缺省使用全局单例).
            pipeline: L7 渲染流水线 (可选, 缺省惰性创建).
            cors_origins: CORS 允许的源 (默认 ["*"]).
            api_prefix: API 版本前缀 (如 "/api/v1", 默认 "" 无前缀).
        """
        self._am = artifact_manager if artifact_manager is not None else ArtifactManager()
        self._registry = registry if registry is not None else get_registry()
        self._pipeline = _resolve_pipeline(pipeline, self._registry, self._am)
        self._cors_origins = cors_origins if cors_origins is not None else ["*"]
        self._api_prefix = api_prefix.rstrip("/") if api_prefix else ""
        self._handlers = _RouteHandlers(
            artifact_manager=self._am,
            registry=self._registry,
            pipeline=self._pipeline,
        )

    def create_app(self) -> Starlette:
        """创建 Starlette 应用实例."""
        h = self._handlers
        p = self._api_prefix

        # 基础路由 (无前缀)
        base_routes = [
            Route("/health", h.health, methods=["GET"]),
            Route("/render", h.render, methods=["POST"]),
            Route("/render/{artifact_id}", h.get_cached_render, methods=["GET"]),
            Route("/render/{artifact_id}", h.update_render, methods=["PUT"]),
            Route("/render/{artifact_id}", h.destroy_render, methods=["DELETE"]),
            Route("/artifacts", h.list_artifacts, methods=["GET"]),
            Route("/artifacts/{artifact_id}", h.get_artifact, methods=["GET"]),
            Route("/artifacts/{artifact_id}/versions", h.get_versions, methods=["GET"]),
            Route("/artifacts/{artifact_id}/edit", h.edit_artifact, methods=["POST"]),
            Route("/registry/mime-types", h.list_mime_types, methods=["GET"]),
            Route("/registry/renderers", h.list_renderers, methods=["GET"]),
            Route("/registry/supports/{mime_type:path}", h.supports_mime, methods=["GET"]),
            Route("/stats", h.stats, methods=["GET"]),
            # OpenAPI 文档端点
            Route("/openapi.json", openapi_handler(self), methods=["GET"]),
            Route("/docs", swagger_ui_handler(self), methods=["GET"]),
        ]

        routes = list(base_routes)

        # 版本前缀路由 (如果有前缀, 所有端点在前缀下也可用)
        if p:
            prefixed_routes = [
                Route(f"{p}/health", h.health, methods=["GET"]),
                Route(f"{p}/render", h.render, methods=["POST"]),
                Route(f"{p}/render/{{artifact_id}}", h.get_cached_render, methods=["GET"]),
                Route(f"{p}/render/{{artifact_id}}", h.update_render, methods=["PUT"]),
                Route(f"{p}/render/{{artifact_id}}", h.destroy_render, methods=["DELETE"]),
                Route(f"{p}/artifacts", h.list_artifacts, methods=["GET"]),
                Route(f"{p}/artifacts/{{artifact_id}}", h.get_artifact, methods=["GET"]),
                Route(f"{p}/artifacts/{{artifact_id}}/versions", h.get_versions, methods=["GET"]),
                Route(f"{p}/artifacts/{{artifact_id}}/edit", h.edit_artifact, methods=["POST"]),
                Route(f"{p}/registry/mime-types", h.list_mime_types, methods=["GET"]),
                Route(f"{p}/registry/renderers", h.list_renderers, methods=["GET"]),
                Route(f"{p}/registry/supports/{{mime_type:path}}", h.supports_mime, methods=["GET"]),
                Route(f"{p}/stats", h.stats, methods=["GET"]),
                # OpenAPI 文档端点 (前缀版本)
                Route(f"{p}/openapi.json", openapi_handler(self), methods=["GET"]),
                Route(f"{p}/docs", swagger_ui_handler(self), methods=["GET"]),
            ]
            routes.extend(prefixed_routes)

        middleware = []
        if self._cors_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=self._cors_origins,
                    allow_methods=["*"] if "*" in self._cors_origins
                                   else ["GET", "POST", "PUT", "DELETE"],
                    allow_headers=["*"],
                )
            )

        app = Starlette(routes=routes, middleware=middleware)
        return app

    def get_routes_summary(self) -> list[dict[str, Any]]:
        """获取所有路由摘要 (用于文档/发现).

        当设置了 ``api_prefix`` 时, 路由路径包含前缀。
        """
        p = self._api_prefix
        base = [
            ("health", "GET", "L7 体验呈现层健康检查"),
            ("render", "POST", "渲染 Artifact"),
            ("render/{artifact_id}", "GET", "获取缓存的渲染描述"),
            ("render/{artifact_id}", "PUT", "增量更新渲染"),
            ("render/{artifact_id}", "DELETE", "销毁渲染实例"),
            ("artifacts", "GET", "Artifact 列表 (支持 query 过滤 + 分页排序)"),
            ("artifacts/{artifact_id}", "GET", "Artifact 详情"),
            ("artifacts/{artifact_id}/versions", "GET", "版本历史"),
            ("artifacts/{artifact_id}/edit", "POST", "提交编辑 (RFC 6902 JSON Patch)"),
            ("registry/mime-types", "GET", "已注册 MIME 类型列表"),
            ("registry/renderers", "GET", "已注册渲染器列表"),
            ("registry/supports/{mime_type}", "GET", "检查 MIME 支持"),
            ("stats", "GET", "统计信息 (pipeline + artifact)"),
        ]
        return [
            {"path": f"{p}/{path}", "methods": [method], "description": desc}
            for path, method, desc in base
        ]


__all__ = ["L7Router"]
