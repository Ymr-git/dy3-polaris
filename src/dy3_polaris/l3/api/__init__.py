"""L3 领域知识层 — REST API 路由包.

基于 Starlette 构建, 将 L3 知识层的完整功能暴露为 RESTful JSON API。
遵循与 L6 API 一致的设计模式: 统一响应格式、CORS 中间件、异常统一处理。

端点概览 (40+ REST 端点):
    # 健康检查
    GET  /l3/health                      — L3 知识层健康检查

    # 知识实体管理 (CRUD)
    POST /l3/entities                    — 创建知识实体
    GET  /l3/entities                    — 列出知识实体 (支持分页/类型过滤)
    GET  /l3/entities/{id}               — 获取单个实体
    PUT  /l3/entities/{id}               — 更新实体
    DELETE /l3/entities/{id}             — 删除实体

    # 三元组管理
    POST /l3/triples                     — 创建三元组
    GET  /l3/triples                     — 查询三元组 (按主语/谓词/宾语)
    DELETE /l3/triples/{id}              — 删除三元组

    # 知识检索
    POST /l3/retrieve/keyword            — 关键词检索
    POST /l3/retrieve/vector             — 向量检索
    POST /l3/retrieve/hybrid             — 混合检索
    POST /l3/retrieve/intent             — 意图驱动路由检索

    # 知识摄入
    POST /l3/ingest                      — 知识摄入管道 (分块→分类→验证→去重→存储)
    POST /l3/ingest/batch                — 批量摄入

    # 事实校验
    POST /l3/fact-check                  — 事实校验
    GET  /l3/standards                   — 获取标准值列表
    POST /l3/standards                   — 添加标准值

    # 质量管理
    POST /l3/quality/assess              — 单实体质量评估 (六维)
    POST /l3/quality/assess/batch        — 批量质量评估
    POST /l3/quality/assess/global       — 全库质量评估
    POST /l3/quality/conflicts/detect    — 冲突检测
    POST /l3/quality/conflicts/resolve   — 冲突消解
    GET  /l3/quality/dashboard           — 质量仪表板
    GET  /l3/quality/provenance/{id}     — 溯源查询
    POST /l3/quality/provenance          — 记录溯源
    GET  /l3/quality/audit-log           — 审计日志

    # 图推理
    POST /l3/graph/reason                — 图推理 (路径/多跳/规则/链接预测)
    GET  /l3/graph/stats                 — 图统计

    # 本体管理
    GET  /l3/ontology/domains            — 列出所有领域
    GET  /l3/ontology/{domain}           — 获取领域本体
    POST /l3/ontology/validate           — 本体验证

    # 持久化
    POST /l3/persistence/snapshot        — 保存快照
    POST /l3/persistence/restore         — 恢复快照

    # 知识库统计
    GET  /l3/stats                       — 知识库统计信息
"""

from .router import L3Router

__all__ = ["L3Router"]
