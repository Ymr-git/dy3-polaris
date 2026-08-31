"""L7 体验呈现层 — 基础架构与 IRenderer 接口.

L7 是 7 层 + 3 横切架构中唯一面向用户感知的前端层，负责将
L0-L6 全部后端能力转化为可视化、可交互、可理解的最终用户体验。

核心组件:
- IRenderer: 渲染器统一抽象接口
- RendererRegistry: MIME type 路由的渲染器注册中心
- ArtifactManager: Artifact 全生命周期管理 (DAG 版本树)
- RenderPipeline: 渲染流水线 (缓存/增量更新/批量/懒加载)
- L7Router: RESTful API 路由

融合世界先进方案:
- Jupyter nbformat: MIME Bundle 多模态渲染
- VS Code CustomEditor: 统一编辑器生命周期
- React Server Components: 增量更新 + 资源管理
- Git DAG: 版本树管理
- IntersectionObserver: 视口懒加载
"""
from .exceptions import (
    L7Error, RendererNotFoundError, ArtifactNotFoundError,
    ArtifactValidationError, RenderTimeoutError, UnsupportedMimeError,
    VersionConflictError, ArtifactNotEditableError, RenderContextError,
)
from .models import (
    ArtifactType, ArtifactLifecycleState, LearnerMode, DiffOpType,
    Artifact, DiffOp, ArtifactDiff, RenderContext, Viewport,
    RenderDescriptor, ArtifactVersionNode, VersionTree,
    MIME_TO_TYPE, TYPE_TO_MIME,
)
from .irenderer import IRenderer
from .registry import RendererRegistry, get_registry, reset_registry
from .artifact_manager import ArtifactManager
from .events import (
    L7Event, EventEmitter, get_global_emitter, reset_global_emitter,
    RENDER_START, RENDER_SUCCESS, RENDER_ERROR,
    UPDATE_START, UPDATE_SUCCESS, UPDATE_ERROR,
    DESTROY_START, DESTROY_SUCCESS,
    ARTIFACT_REGISTERED, ARTIFACT_UPDATED, ARTIFACT_REMOVED,
    ARTIFACT_REVIEWED, ARTIFACT_ARCHIVED, ARTIFACT_RESTORED, ARTIFACT_MERGED,
)

__all__ = [
    # Exceptions
    "L7Error", "RendererNotFoundError", "ArtifactNotFoundError",
    "ArtifactValidationError", "RenderTimeoutError", "UnsupportedMimeError",
    "VersionConflictError", "ArtifactNotEditableError", "RenderContextError",
    # Enums
    "ArtifactType", "ArtifactLifecycleState", "LearnerMode", "DiffOpType",
    # Models
    "Artifact", "DiffOp", "ArtifactDiff", "RenderContext", "Viewport",
    "RenderDescriptor", "ArtifactVersionNode", "VersionTree",
    "MIME_TO_TYPE", "TYPE_TO_MIME",
    # Core
    "IRenderer", "RendererRegistry", "get_registry", "reset_registry",
    "ArtifactManager",
    # Events
    "L7Event", "EventEmitter", "get_global_emitter", "reset_global_emitter",
    "RENDER_START", "RENDER_SUCCESS", "RENDER_ERROR",
    "UPDATE_START", "UPDATE_SUCCESS", "UPDATE_ERROR",
    "DESTROY_START", "DESTROY_SUCCESS",
    "ARTIFACT_REGISTERED", "ARTIFACT_UPDATED", "ARTIFACT_REMOVED",
    "ARTIFACT_REVIEWED", "ARTIFACT_ARCHIVED", "ARTIFACT_RESTORED", "ARTIFACT_MERGED",
]

# BaseRenderer / FallbackRenderer 渲染器基类与降级渲染器, 条件导出
try:
    from .base_renderer import BaseRenderer, FallbackRenderer
    __all__.extend(["BaseRenderer", "FallbackRenderer"])
except ImportError:
    pass

# RenderPipeline 由另一代理并行实现, 条件导出 (尚未实现时静默跳过)
try:
    from .pipeline import RenderPipeline
    __all__.append("RenderPipeline")
except ImportError:
    pass

# L7Router RESTful API 路由, 条件导出
try:
    from .api import L7Router
    __all__.append("L7Router")
except ImportError:
    pass

# 渲染钩子 / 中间件系统 (Render Hooks / Middleware), 条件导出
try:
    from .hooks import (
        RenderHook,
        BaseRenderHook,
        HookRegistry,
        HookablePipeline,
        LoggingHook,
        TimingHook,
        CachingHook,
    )
    __all__.extend([
        "RenderHook",
        "BaseRenderHook",
        "HookRegistry",
        "HookablePipeline",
        "LoggingHook",
        "TimingHook",
        "CachingHook",
    ])
except ImportError:
    pass

# 学情面板与多模态输出 (任务拆分 T4) — BKT/进度/辩论/参数调节器/虚拟实验台/图谱探索器
try:
    from .dashboard import (
        render_bkt_dashboard,
        render_kp_detail,
        render_progress_panel,
        render_debate,
        render_drill_down,
        render_time_travel,
        render_comparison,
    )
    from .interactive import (
        render_param_controller,
        render_virtual_lab,
        render_graph_explorer,
    )
    __all__.extend([
        "render_bkt_dashboard",
        "render_kp_detail",
        "render_progress_panel",
        "render_debate",
        "render_drill_down",
        "render_time_travel",
        "render_comparison",
        "render_param_controller",
        "render_virtual_lab",
        "render_graph_explorer",
    ])
except ImportError:
    pass

# 溯源可视化与 CC2 审批 (任务拆分 T5) — 溯源时间线/决策溯源/贡献图谱/分支合并 + 审批面板
try:
    from .provenance import (
        render_timeline,
        render_decision_trace,
        render_agent_contribution,
        render_branch_merge,
    )
    from .approval import (
        render_plan_preview,
        render_approval_flow,
        render_quick_mode,
        render_plan_rendering,
    )
    __all__.extend([
        "render_timeline",
        "render_decision_trace",
        "render_agent_contribution",
        "render_branch_merge",
        "render_plan_preview",
        "render_approval_flow",
        "render_quick_mode",
        "render_plan_rendering",
    ])
except ImportError:
    pass

# 响应式/无障碍/国际化/性能监控 (任务拆分 T6)
try:
    from .responsive import (
        breakpoint_for,
        layout_plan,
        render_layout,
    )
    from .accessibility import (
        contrast_ratio,
        passes_aa,
        audit_contrast,
        a11y_attributes,
        high_contrast_css,
        colorblind_css,
        ishihara_test,
    )
    from .i18n import (
        translate,
        glossary,
        i18n_init_config,
        normalize_locale,
        format_number,
        format_date,
    )
    from .monitoring import (
        PerformanceTracker,
        check_vitals,
        check_render_latency,
    )
    __all__.extend([
        "breakpoint_for", "layout_plan", "render_layout",
        "contrast_ratio", "passes_aa", "audit_contrast", "a11y_attributes",
        "high_contrast_css", "colorblind_css", "ishihara_test",
        "translate", "glossary", "i18n_init_config", "normalize_locale",
        "format_number", "format_date",
        "PerformanceTracker", "check_vitals", "check_render_latency",
    ])
except ImportError:
    pass

# 七大 Native Renderer (任务拆分 T2) — 文本/图表/图谱/分子/表格/公式/溯源
try:
    from .renderers import (
        TextRenderer,
        MarkdownRenderer,
        ChartRenderer,
        GraphRenderer,
        MoleculeRenderer,
        TableRenderer,
        FormulaRenderer,
        ProvenanceRenderer,
        register_native_renderers,
        native_renderer_classes,
    )
    __all__.extend([
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
    ])
except ImportError:
    pass

# Artifact 管理系统 (任务拆分 T3) — 生命周期/版本DAG/三级存储/搜索/编辑通道
try:
    from .artifact import (
        LifecycleStateMachine,
        StateTransitionError,
        ArtifactVersionGraph,
        MergeConflictError,
        MergeResult,
        VersionNode,
        TieredArtifactStore,
        MemoryArtifactStore,
        JsonFileArtifactStore,
        ServerArtifactStore,
        NoopServerStore,
        ArtifactStore,
        ContentStore,
        SearchEngine,
        InvertedIndex,
        ArtifactMetadata,
        EditPermission,
    )
    __all__.extend([
        "LifecycleStateMachine",
        "StateTransitionError",
        "ArtifactVersionGraph",
        "MergeConflictError",
        "MergeResult",
        "VersionNode",
        "TieredArtifactStore",
        "MemoryArtifactStore",
        "JsonFileArtifactStore",
        "ServerArtifactStore",
        "NoopServerStore",
        "ArtifactStore",
        "ContentStore",
        "SearchEngine",
        "InvertedIndex",
        "ArtifactMetadata",
        "EditPermission",
    ])
except ImportError:
    pass
