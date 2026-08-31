"""响应契约单点 (SSOT) — 错误码注册表 + 统一响应信封.

设计:
- 错误码注册表: 集中登记全系统负数码, 新码必须经 register_error_code
  (重复分配即报错, 消除跨层码段撞车)
- 响应信封: ok()/err() 单一构造器, err 自动回填请求级 trace_id
- 各层 _ok/_err 均为本模块 re-export (保留原调用签名兼容)
"""
from __future__ import annotations

import threading
from typing import Any

from dy3_polaris.l5.tracing import get_trace_id

# ---------------------------------------------------------------------------
# 1. 错误码注册表
# ---------------------------------------------------------------------------

#: code -> {name, message, http_status}
_ERROR_CODES: dict[int, dict[str, Any]] = {}
_ERR_LOCK = threading.RLock()


def register_error_code(
    code: int,
    name: str,
    default_message: str,
    http_status: int = 400,
) -> None:
    """登记错误码 (单点分配; 重复分配抛 ValueError)."""
    with _ERR_LOCK:
        existing = _ERROR_CODES.get(code)
        if existing and existing["name"] != name:
            raise ValueError(
                f"错误码 {code} 重复分配: {existing['name']} (层 {existing.get('owner','?')}) "
                f"vs {name}; 请改用未占用码段"
            )
        _ERROR_CODES[code] = {
            "name": name,
            "message": default_message,
            "http_status": http_status,
            "owner": existing["owner"] if existing else name.split(".")[0],
        }


def error_code_info(code: int) -> dict[str, Any] | None:
    """查询错误码元信息."""
    return _ERROR_CODES.get(code)


def registered_error_codes() -> dict[int, dict[str, Any]]:
    """全量注册表 (调试/展示)."""
    return dict(_ERROR_CODES)


# ---------------------------------------------------------------------------
# 2. 内置注册 (JSON-RPC 保留段 + 跨层通用码)
# ---------------------------------------------------------------------------

# 公共段 (-32700 ~ -32000): JSON-RPC 协议保留
register_error_code(-32700, "PARSE_ERROR", "Parse error", 400)
register_error_code(-32701, "INVALID_REQUEST", "Invalid request", 400)
register_error_code(-32702, "METHOD_NOT_FOUND", "Method not found", 404)
register_error_code(-32703, "IDEMPOTENCY_ERROR", "Duplicate request", 409)
register_error_code(-32600, "NOT_FOUND", "Resource not found", 404)
register_error_code(-32601, "TOOL_NOT_FOUND", "Tool not found", 404)
register_error_code(-32602, "INVALID_PARAMS", "Invalid params", 400)
register_error_code(-32603, "INTERNAL_ERROR", "Internal Server Error", 500)

# L1 认证/会话段 (-32200 ~ -32209)
register_error_code(-32200, "AUTH_ERROR", "Authentication error", 401)
register_error_code(-32201, "AUTHENTICATION_ERROR", "Authentication required", 401)
register_error_code(-32202, "TOKEN_ERROR", "Invalid token", 401)
register_error_code(-32203, "FORBIDDEN", "Permission denied", 403)
register_error_code(-32205, "LIFECYCLE_ERROR", "Lifecycle error", 409)
register_error_code(-32206, "SESSION_EXPIRED", "Session expired", 401)

# L2 学习域段 (-32300 ~ -32319)
register_error_code(-32300, "LEARNING_ERROR", "Learning operation failed", 400)
register_error_code(-32310, "PROFILE_CONFLICT", "Profile version conflict", 409)

# 跨层通用段 (-32400 ~ -32499)
register_error_code(-32400, "OPERATION_FAILED", "Operation failed", 500)
register_error_code(-32401, "SERVICE_UNAVAILABLE", "Service unavailable", 503)
register_error_code(-32501, "SESSION_NOT_FOUND", "Session not found", 404)
register_error_code(-32502, "CACHE_NOT_FOUND", "Render cache not found", 404)
register_error_code(-32503, "FORK_ERROR", "Session fork failed", 400)
register_error_code(-32504, "ARTIFACT_ERROR", "Artifact error", 400)

# L6 协议段 (-32000 ~ -32099)
register_error_code(-32000, "L6_PROTOCOL_ERROR", "Protocol error", 400)

# ---------------------------------------------------------------------------
# 3. 统一响应信封
# ---------------------------------------------------------------------------


def ok(data: Any = None, message: str = "") -> dict[str, Any]:
    """构造成功响应 (统一 {code:0, data, message})."""
    return {"code": 0, "data": data, "message": message}


def err(
    code: int,
    message: str,
    detail: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """构造错误响应 (统一 {code, message, detail?, trace_id?, ...}).

    - detail: 附加详情 (非空时写入)
    - trace_id: 自动回填请求级 trace_id (可显式覆盖)
    - extra: 业务扩展字段 (如 PROFILE_CONFLICT 的 current_version)
    """
    resp: dict[str, Any] = {"code": code, "message": message}
    if detail:
        resp["detail"] = detail
    tid = extra.pop("trace_id", None) or get_trace_id()
    if tid:
        resp["trace_id"] = tid
    resp.update(extra)
    return resp


# 兼容别名 (旧调用方按 _ok/_err 引用)
_ok = ok
_err = err

__all__ = [
    "err",
    "error_code_info",
    "ok",
    "register_error_code",
    "registered_error_codes",
]
