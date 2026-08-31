"""L7 API — 错误码定义 (error_codes.py).

任务拆分 T6 · 设计文档 Ch.9.5。

8 个业务错误码 + HTTP 状态码映射 + 统一错误格式。
"""

from __future__ import annotations

from typing import Any

#: 业务错误码定义 (Ch.9.5)
ERROR_CODES: dict[str, dict[str, Any]] = {
    "RENDER_UNSUPPORTED_MIME": {
        "code": "RENDER_UNSUPPORTED_MIME",
        "http_status": 422,
        "message": "不支持的 MIME type",
        "description": "Agent 产出 L7 无法渲染的 Artifact 类型",
    },
    "RENDER_PAYLOAD_INVALID": {
        "code": "RENDER_PAYLOAD_INVALID",
        "http_status": 422,
        "message": "payload 结构校验失败",
        "description": "图表数据缺少必要 X/Y 轴定义",
    },
    "ARTIFACT_NOT_FOUND": {
        "code": "ARTIFACT_NOT_FOUND",
        "http_status": 404,
        "message": "Artifact 不存在",
        "description": "引用了已过期的 artifact_id",
    },
    "ARTIFACT_READONLY": {
        "code": "ARTIFACT_READONLY",
        "http_status": 403,
        "message": "Artifact 不可编辑",
        "description": "尝试编辑 CC2 已批准的教学内容",
    },
    "EDIT_REJECTED": {
        "code": "EDIT_REJECTED",
        "http_status": 409,
        "message": "编辑被 Agent 拒绝",
        "description": "Agent 判断编辑建议不合理",
    },
    "SESSION_EXPIRED": {
        "code": "SESSION_EXPIRED",
        "http_status": 401,
        "message": "会话已过期",
        "description": "JWT token 超过有效期",
    },
    "RATE_LIMITED": {
        "code": "RATE_LIMITED",
        "http_status": 429,
        "message": "请求频率超限",
        "description": "渲染请求超过 30 次/分钟",
    },
    "DASHBOARD_NO_DATA": {
        "code": "DASHBOARD_NO_DATA",
        "http_status": 404,
        "message": "无学情数据",
        "description": "新用户尚未开始学习",
    },
}


def error_payload(code: str, trace_id: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    """生成统一错误响应 (Ch.9.4 callout).

    Args:
        code: 业务错误码。
        trace_id: 追踪 ID (写入 L0 Audit Trail)。
        details: 附加错误详情。

    Returns:
        统一错误格式 {"status", "code", "message", "trace_id", "details"}。
    """
    meta = ERROR_CODES.get(code, {
        "code": code,
        "http_status": 500,
        "message": "未知错误",
        "description": "",
    })
    # 请求级 trace_id 回填: 显式传参优先, 否则取中间件注入的上下文
    from dy3_polaris.l5.tracing import get_trace_id

    return {
        "status": "error",
        "code": meta["code"],
        "message": meta["message"],
        "trace_id": trace_id or get_trace_id(),
        "details": details or {},
    }


def http_status_for(code: str) -> int:
    """返回错误码对应的 HTTP 状态码."""
    meta = ERROR_CODES.get(code)
    return meta["http_status"] if meta else 500


def all_error_codes() -> dict[str, dict[str, Any]]:
    """返回全部错误码定义 (API 文档用)."""
    return {k: dict(v) for k, v in ERROR_CODES.items()}
