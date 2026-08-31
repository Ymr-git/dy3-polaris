"""L7 体验呈现层 — 渲染流水线 (RenderPipeline).

编排 Artifact → RendererRegistry → IRenderer → RenderDescriptor 的完整渲染流程，
提供缓存、增量更新、批量渲染、视口懒加载和超时控制。

融合世界先进方案:
- React Server Components: 增量更新 + 资源生命周期管理
- IntersectionObserver: 视口可见性驱动的懒加载
- Jupyter nbconvert: 批量渲染管道
- Grafana plugin: 渲染缓存 + TTL 过期
- concurrent.futures: 真正的渲染超时中断 (替代事后检查)

增强特性 (向后兼容):
1. 缓存 TTL — ``cache_ttl_seconds`` 控制缓存项过期；``get_cache_age()`` /
   ``evict_expired()`` 提供过期管理。TTL=0 (默认) 时永不过期，与原行为一致。
2. 上下文感知缓存键 — ``_cache_key(artifact_id, version, context)`` 生成
   ``"{artifact_id}:v{version}:h{context_hash}"`` 格式键；version=0 且
   context=None 时退化为简单 ``artifact_id``，确保向后兼容。
3. 真超时 — 使用 ``concurrent.futures.ThreadPoolExecutor`` 在独立线程中
   执行渲染，``future.result(timeout=...)`` 实现真正的超时中断。

核心流程:
    Artifact → render(artifact, context)
             → registry.get_renderer_for_artifact(artifact)
             → renderer.render(artifact, context)
             → RenderDescriptor (缓存 + 状态更新)

    ArtifactDiff → update(artifact_id, diff)
                 → renderer.update(diff)
                 → RenderDescriptor (缓存更新)

线程安全: 使用 threading.RLock 保护所有公开方法。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import threading
import time
from typing import Any

from .artifact_manager import ArtifactManager
from .exceptions import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    L7Error,
    RenderContextError,
    RenderTimeoutError,
    UnsupportedMimeError,
)
from .irenderer import IRenderer
from .models import (
    Artifact,
    ArtifactDiff,
    ArtifactLifecycleState,
    RenderContext,
    RenderDescriptor,
)
from .registry import RendererRegistry, get_registry

_logger = logging.getLogger("dy3_polaris.l7.pipeline")


class RenderPipeline:
    """渲染流水线 — 编排 Artifact 到 RenderDescriptor 的完整渲染流程.

    融合方案:
    - React Server Components: 增量更新 (update) + 资源生命周期 (destroy)
    - IntersectionObserver: 视口可见性驱动的懒加载 (render_if_visible / mark_visible / mark_hidden)
    - Jupyter nbconvert: 批量渲染管道 (render_batch)
    - Grafana plugin: 渲染缓存 + TTL 过期 + 统计 (cache + get_stats)
    - concurrent.futures: 真正的渲染超时中断

    线程安全: 使用 threading.RLock 保护所有操作。

    Attributes:
        _registry: 渲染器注册中心 (MIME → IRenderer 路由)
        _artifact_manager: Artifact 生命周期管理器
        _lock: 可重入线程锁
        _cache_ttl_seconds: 缓存过期时间 (秒)，0 表示永不过期
        _cache: cache_key → (RenderDescriptor, timestamp) 缓存项
        _renderer_instances: cache_key → 活跃渲染器实例 (用于 update)
        _visible: 当前可见的 artifact ID 集合
        _stats: 渲染统计信息
    """

    def __init__(
        self,
        registry: RendererRegistry | None = None,
        artifact_manager: ArtifactManager | None = None,
        cache_ttl_seconds: float = 0,
    ) -> None:
        """初始化 RenderPipeline.

        Args:
            registry: 渲染器注册中心，默认使用全局单例.
            artifact_manager: Artifact 管理器，默认创建新实例.
            cache_ttl_seconds: 缓存过期时间 (秒)。0 (默认) 表示永不过期，
                保持与原实现完全一致的行为；> 0 时缓存项在超过该时长后
                自动失效 (get_cached/render 视为未命中，evict_expired 主动清除).
        """
        self._registry: RendererRegistry = (
            registry if registry is not None else get_registry()
        )
        self._artifact_manager: ArtifactManager = (
            artifact_manager if artifact_manager is not None else ArtifactManager()
        )
        self._lock = threading.RLock()

        # 缓存 TTL (0 = 永不过期)
        self._cache_ttl_seconds: float = float(cache_ttl_seconds)

        # 内部状态: cache_key → (RenderDescriptor, 存储时间戳)
        self._cache: dict[str, tuple[RenderDescriptor, float]] = {}
        self._renderer_instances: dict[str, IRenderer] = {}
        self._visible: set[str] = set()

        # 统计
        self._stats: dict[str, Any] = {
            "total_renders": 0,
            "cache_hits": 0,
            "total_render_time_ms": 0.0,
        }

    # ============================================================
    # 缓存键与过期辅助 (上下文感知缓存键 + TTL)
    # ============================================================

    def _cache_key(
        self,
        artifact_id: str,
        version: int = 0,
        context: RenderContext | None = None,
    ) -> str:
        """生成上下文感知缓存键.

        键格式: ``"{artifact_id}:v{version}:h{context_hash}"``，其中 context_hash
        由 (theme, learner_mode, viewport.width, viewport.height, locale) 计算。

        向后兼容: version=0 且 context=None 时退化为简单 ``artifact_id``，
        确保不传 version/context 的现有调用与 get_cached(artifact_id) 行为一致。

        Args:
            artifact_id: Artifact ID.
            version: 缓存版本号 (0 = 不纳入键).
            context: 渲染上下文 (None = 不纳入键).

        Returns:
            缓存键字符串.
        """
        if version == 0 and context is None:
            return artifact_id

        parts: list[str] = [artifact_id]
        if version != 0:
            parts.append(f"v{version}")
        if context is not None:
            parts.append(f"h{self._context_hash(context)}")
        return ":".join(parts)

    @staticmethod
    def _context_hash(context: RenderContext) -> str:
        """计算渲染上下文的稳定哈希 (8 位十六进制).

        哈希输入: (theme, learner_mode, viewport.width, viewport.height, locale).
        使用 hashlib.sha256 保证跨进程/跨运行稳定 (不依赖 PYTHONHASHSEED).
        """
        mode = context.learner_mode
        mode_val = mode.value if hasattr(mode, "value") else str(mode)
        key_tuple = (
            context.theme,
            mode_val,
            context.viewport.width,
            context.viewport.height,
            context.locale,
        )
        digest = hashlib.sha256(repr(key_tuple).encode("utf-8")).hexdigest()
        return digest[:8]

    def _is_expired(self, timestamp: float) -> bool:
        """判断给定时间戳的缓存项是否已过期.

        TTL=0 时永不过期 (返回 False).
        """
        if self._cache_ttl_seconds <= 0:
            return False
        return (time.time() - timestamp) > self._cache_ttl_seconds

    def _get_cached_entry(self, cache_key: str) -> RenderDescriptor | None:
        """获取未过期的缓存项；不存在或已过期均返回 None."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        descriptor, timestamp = entry
        if self._is_expired(timestamp):
            return None
        return descriptor

    # ============================================================
    # 核心渲染
    # ============================================================

    def render(
        self,
        artifact: Artifact,
        context: RenderContext,
        force: bool = False,
        timeout_ms: int = 30000,
        version: int = 0,
    ) -> RenderDescriptor:
        """渲染 Artifact 为 RenderDescriptor.

        完整流程:
        1. 检查缓存 (非 force 时命中且未过期则直接返回)
        2. 校验 Artifact (artifact.validate())
        3. 从注册中心获取渲染器
        4. 执行渲染 (真超时: concurrent.futures 线程 + future.result(timeout))
        5. 缓存描述符 + 存储渲染器实例
        6. 更新 Artifact 状态为 RENDERED
        7. 更新统计

        Args:
            artifact: 待渲染的制品.
            context: 渲染上下文.
            force: 是否强制重新渲染 (忽略缓存).
            timeout_ms: 超时时间 (毫秒)，默认 30000.
            version: 缓存版本号 (0 = 退化为简单 artifact_id 键，向后兼容；
                > 0 时启用上下文感知缓存键，包含 version 与 context 哈希).

        Returns:
            RenderDescriptor: 前端可消费的渲染描述符.

        Raises:
            ArtifactValidationError: Artifact 校验失败.
            UnsupportedMimeError: 没有注册对应 MIME 的渲染器.
            RenderTimeoutError: 渲染超时.
        """
        with self._lock:
            artifact_id = artifact.artifact_id

            # 上下文感知缓存键: version=0 时退化为简单 artifact_id (向后兼容)
            if version == 0:
                cache_key = self._cache_key(artifact_id)
            else:
                cache_key = self._cache_key(artifact_id, version, context)

            # 1. 检查缓存 (含 TTL 过期检查)
            if not force:
                cached = self._get_cached_entry(cache_key)
                if cached is not None:
                    self._stats["cache_hits"] += 1
                    _logger.debug("Cache hit for artifact %s", artifact_id)
                    return cached

            # 2. 校验 Artifact
            artifact.validate()

            # 3. 获取渲染器
            if not self._registry.is_supported(artifact.mime):
                raise UnsupportedMimeError(artifact.mime)
            renderer = self._registry.get_renderer_for_artifact(artifact)

            # 4. 执行渲染 (真超时: concurrent.futures)
            start = time.time()
            timeout_seconds = timeout_ms / 1000.0
            try:
                descriptor = self._render_with_timeout(
                    renderer, artifact, context, timeout_seconds
                )
            except concurrent.futures.TimeoutError:
                _logger.warning(
                    "Render timed out for artifact %s (limit: %dms)",
                    artifact_id,
                    timeout_ms,
                )
                raise RenderTimeoutError(
                    timeout_seconds=timeout_seconds,
                    detail=(
                        f"Render timed out after {timeout_ms}ms "
                        f"for artifact {artifact_id}"
                    ),
                )
            elapsed_ms = (time.time() - start) * 1000.0

            # 5. 记录渲染耗时
            descriptor.render_time_ms = elapsed_ms

            # 6. 缓存描述符 + 存储渲染器实例 (携带存储时间戳用于 TTL)
            self._cache[cache_key] = (descriptor, time.time())
            self._renderer_instances[cache_key] = renderer

            # 7. 更新 Artifact 状态为 RENDERED
            artifact.state = ArtifactLifecycleState.RENDERED
            if self._artifact_manager is not None:
                try:
                    managed = self._artifact_manager.get(artifact_id)
                    managed.state = ArtifactLifecycleState.RENDERED
                except ArtifactNotFoundError:
                    pass  # Artifact 不在管理器中，跳过

            # 8. 更新统计
            self._stats["total_renders"] += 1
            self._stats["total_render_time_ms"] += elapsed_ms

            _logger.debug(
                "Rendered artifact %s in %.2fms", artifact_id, elapsed_ms
            )
            return descriptor

    def _render_with_timeout(
        self,
        renderer: IRenderer,
        artifact: Artifact,
        context: RenderContext,
        timeout_seconds: float,
    ) -> RenderDescriptor:
        """在独立线程中执行渲染，实现真正的超时中断.

        使用 ``concurrent.futures.ThreadPoolExecutor(max_workers=1)`` 提交渲染
        任务，``future.result(timeout=...)`` 在超时后立即抛出
        ``concurrent.futures.TimeoutError``。超时后调用 ``shutdown(wait=False)``
        不再等待被遗弃的工作线程 (Python 线程无法被强制中断，工作线程会继续
        执行直到自然完成)。

        Args:
            renderer: 渲染器实例.
            artifact: 待渲染的制品.
            context: 渲染上下文.
            timeout_seconds: 超时时间 (秒).

        Returns:
            渲染描述符.

        Raises:
            concurrent.futures.TimeoutError: 渲染超时.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(renderer.render, artifact, context)
        try:
            return future.result(timeout=timeout_seconds)
        finally:
            # 超时时不等待被遗弃的工作线程；正常完成时同样不阻塞
            executor.shutdown(wait=False)

    # ============================================================
    # 增量更新
    # ============================================================

    def update(
        self,
        artifact_id: str,
        diff: ArtifactDiff,
    ) -> RenderDescriptor:
        """增量更新已渲染的 Artifact.

        基于 ArtifactDiff 调用渲染器的 update() 方法，避免全量重渲染。
        借鉴 React Server Components 的增量更新模型。

        Args:
            artifact_id: 要更新的 Artifact ID.
            diff: Artifact 增量差异.

        Returns:
            RenderDescriptor: 更新后的渲染描述符.

        Raises:
            ArtifactNotFoundError: Artifact 未在管理器中注册.
            RenderContextError: 没有先前的渲染实例 (需要先 render()).
        """
        with self._lock:
            # 1. 检查 Artifact 是否在管理器中注册
            if self._artifact_manager is not None:
                try:
                    self._artifact_manager.get(artifact_id)
                except ArtifactNotFoundError:
                    raise ArtifactNotFoundError(artifact_id)

            # 2. 获取缓存的渲染器实例
            cache_key = self._cache_key(artifact_id)
            renderer = self._renderer_instances.get(cache_key)
            if renderer is None:
                raise RenderContextError(
                    "renderer_instance",
                    detail=(
                        f"No prior render exists for artifact {artifact_id}; "
                        f"call render() before update()"
                    ),
                )

            # 3. 执行增量更新
            start = time.time()
            descriptor = renderer.update(diff)
            elapsed_ms = (time.time() - start) * 1000.0
            descriptor.render_time_ms = elapsed_ms

            # 4. 更新缓存
            self._cache[cache_key] = (descriptor, time.time())

            _logger.debug(
                "Updated artifact %s in %.2fms", artifact_id, elapsed_ms
            )
            return descriptor

    # ============================================================
    # 生命周期管理
    # ============================================================

    def destroy(self, artifact_id: str) -> None:
        """销毁渲染实例，释放资源.

        调用渲染器的 destroy() 方法，移除缓存和渲染器实例。
        幂等操作 — 对未知 ID 或已销毁的实例不抛异常。

        Args:
            artifact_id: 要销毁的 Artifact ID.
        """
        with self._lock:
            cache_key = self._cache_key(artifact_id)
            renderer = self._renderer_instances.pop(cache_key, None)
            if renderer is not None:
                try:
                    renderer.destroy()
                except Exception:
                    _logger.warning(
                        "Renderer destroy() raised exception for artifact %s",
                        artifact_id,
                        exc_info=True,
                    )
            self._cache.pop(cache_key, None)
            _logger.debug("Destroyed render instance for artifact %s", artifact_id)

    # ============================================================
    # 批量渲染
    # ============================================================

    def render_batch(
        self,
        artifacts: list[Artifact],
        context: RenderContext,
    ) -> tuple[dict[str, RenderDescriptor], dict[str, str]]:
        """批量渲染多个 Artifact.

        借鉴 Jupyter nbconvert 的批量渲染管道，单个失败不影响其他制品。
        返回成功结果和错误信息的元组。

        Args:
            artifacts: 待渲染的制品列表.
            context: 渲染上下文.

        Returns:
            (results, errors) 元组:
            - results: artifact_id → RenderDescriptor (成功的渲染)
            - errors: artifact_id → 错误消息字符串 (失败的渲染)
        """
        results: dict[str, RenderDescriptor] = {}
        errors: dict[str, str] = {}

        for artifact in artifacts:
            try:
                descriptor = self.render(artifact, context)
                results[artifact.artifact_id] = descriptor
            except L7Error as exc:
                errors[artifact.artifact_id] = str(exc.detail) if exc.detail else str(exc)
                _logger.warning(
                    "Batch render failed for artifact %s: %s",
                    artifact.artifact_id,
                    exc,
                )
            except Exception as exc:
                errors[artifact.artifact_id] = str(exc)
                _logger.warning(
                    "Batch render failed for artifact %s: %s",
                    artifact.artifact_id,
                    exc,
                )

        return results, errors

    # ============================================================
    # 视口懒加载 (IntersectionObserver 模式)
    # ============================================================

    def render_if_visible(
        self,
        artifact: Artifact,
        context: RenderContext,
        is_visible: bool,
    ) -> RenderDescriptor | None:
        """根据可见性决定是否渲染.

        借鉴 IntersectionObserver API，仅在制品可见时渲染。
        不可见时跳过渲染以节省资源。

        Args:
            artifact: 待渲染的制品.
            context: 渲染上下文.
            is_visible: 制品是否在视口中可见.

        Returns:
            RenderDescriptor 如果渲染；None 如果不可见被跳过.
        """
        if not is_visible:
            _logger.debug(
                "Skipping render for artifact %s (not visible)",
                artifact.artifact_id,
            )
            return None

        return self.render(artifact, context)

    def mark_visible(self, artifact_id: str) -> None:
        """标记 Artifact 为可见.

        将 artifact_id 加入可见集合。重新渲染在下次 render_if_visible 或
        render 调用时按需触发。

        Args:
            artifact_id: 要标记为可见的 Artifact ID.
        """
        with self._lock:
            self._visible.add(artifact_id)
            _logger.debug("Marked artifact %s as visible", artifact_id)

    def mark_hidden(self, artifact_id: str) -> None:
        """标记 Artifact 为不可见，并释放渲染资源.

        从可见集合移除，并调用 destroy() 释放渲染器实例 (GPU 资源回收)。
        借鉴 IntersectionObserver 的不可见回调 + React 的资源卸载。

        Args:
            artifact_id: 要标记为不可见的 Artifact ID.
        """
        with self._lock:
            self._visible.discard(artifact_id)
            # 销毁渲染器实例以释放 GPU 资源
            self.destroy(artifact_id)
            _logger.debug("Marked artifact %s as hidden (resources released)", artifact_id)

    # ============================================================
    # 缓存管理
    # ============================================================

    def get_cached(
        self,
        artifact_id: str,
        version: int = 0,
        context: RenderContext | None = None,
    ) -> RenderDescriptor | None:
        """获取缓存的渲染描述符.

        支持上下文感知缓存键: 仅传 artifact_id 时使用简单键 (向后兼容)；
        传入 version/context 时使用增强键检索对应缓存项。

        TTL > 0 时，过期缓存视为未命中 (返回 None)。

        Args:
            artifact_id: Artifact ID.
            version: 缓存版本号 (0 = 简单键).
            context: 渲染上下文 (None = 不纳入键).

        Returns:
            缓存的 RenderDescriptor，未缓存或已过期则返回 None.
        """
        with self._lock:
            cache_key = self._cache_key(artifact_id, version, context)
            return self._get_cached_entry(cache_key)

    def get_cache_age(
        self,
        artifact_id: str,
        version: int = 0,
        context: RenderContext | None = None,
    ) -> float | None:
        """获取缓存项的年龄 (秒).

        Args:
            artifact_id: Artifact ID.
            version: 缓存版本号 (0 = 简单键).
            context: 渲染上下文 (None = 不纳入键).

        Returns:
            缓存年龄 (秒)，缓存不存在时返回 None.
        """
        with self._lock:
            cache_key = self._cache_key(artifact_id, version, context)
            entry = self._cache.get(cache_key)
            if entry is None:
                return None
            _, timestamp = entry
            return time.time() - timestamp

    def evict_expired(self) -> int:
        """主动清除所有过期缓存项.

        TTL=0 时永不过期，直接返回 0。

        Returns:
            被清除的缓存项数量.
        """
        with self._lock:
            if self._cache_ttl_seconds <= 0:
                return 0
            now = time.time()
            expired_keys = [
                key
                for key, (_, timestamp) in self._cache.items()
                if (now - timestamp) > self._cache_ttl_seconds
            ]
            for key in expired_keys:
                self._cache.pop(key, None)
            if expired_keys:
                _logger.debug(
                    "Evicted %d expired cache entries", len(expired_keys)
                )
            return len(expired_keys)

    def clear_cache(self) -> None:
        """清空所有缓存的渲染描述符.

        注意: 仅清除缓存，不销毁渲染器实例。
        如需释放渲染器资源，请使用 destroy() 或 mark_hidden()。
        """
        with self._lock:
            self._cache.clear()
            _logger.debug("Cache cleared")

    # ============================================================
    # 统计
    # ============================================================

    def get_stats(self) -> dict[str, Any]:
        """获取渲染统计信息.

        Returns:
            包含以下键的字典:
            - total_renders: 成功渲染次数 (不含缓存命中)
            - cache_size: 当前缓存大小
            - cache_hit_rate: 缓存命中率 (0.0 ~ 1.0)
            - avg_render_time_ms: 平均渲染耗时 (毫秒)
        """
        with self._lock:
            total_renders = self._stats["total_renders"]
            cache_hits = self._stats["cache_hits"]
            total_requests = total_renders + cache_hits

            cache_hit_rate = (
                cache_hits / total_requests if total_requests > 0 else 0.0
            )
            avg_render_time_ms = (
                self._stats["total_render_time_ms"] / total_renders
                if total_renders > 0
                else 0.0
            )

            return {
                "total_renders": total_renders,
                "cache_size": len(self._cache),
                "cache_hit_rate": cache_hit_rate,
                "avg_render_time_ms": avg_render_time_ms,
            }
