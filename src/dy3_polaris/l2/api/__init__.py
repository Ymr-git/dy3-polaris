"""L2 个性化层 — REST API 路由包.

基于 Starlette 构建, 将 L2 个性化层的核心功能暴露为 RESTful JSON API。
遵循与 L3/L6 API 一致的设计模式: 统一响应格式、CORS 中间件、异常统一处理。

端点概览:
    # 健康检查
    GET  /l2/health                      — L2 个性化层健康检查

    # IRT 能力评估 (设计文档 §7.3)
    POST /l2/irt/estimate                — IRT 能力参数估计 (~1次/答题)
    POST /l2/irt/next-question           — CAT 自适应出题 (~1次/出题)
    POST /l2/irt/calibrate               — MMLE 题库参数校准
    GET  /l2/irt/ability/{learner_id}    — 获取能力快照

    # BKT 知识追踪 (设计文档 §7.3)
    POST /l2/bkt/update                  — BKT 单 KP 在线更新 (~1次/答题)

融合世界先进方案 API 设计:
- Knewton API: IRT 驱动能力评估即服务
- ALEKS API: 自适应出题与知识空间查询
- catR / mirt: R 包 CAT 选题与参数校准 REST 化
- Duolingo API: 实时学情更新端点
- OpenAPI 3.0: 资源描述与 schema
- JSON:API spec: 统一响应结构
"""

from .router import L2Router

__all__ = ["L2Router"]
