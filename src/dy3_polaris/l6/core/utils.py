"""L6 核心工具函数."""

from __future__ import annotations

from typing import Any


def snapshot_sanitize(data: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
    """清理快照数据，截断过大的值.

    用于 KPA 输入/输出快照、日志上下文等场景，防止大对象进入溯源链。

    Args:
        data: 原始数据字典
        max_depth: 最大递归深度

    Returns:
        清理后的数据字典
    """
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v if not isinstance(v, str) or len(v) <= 256 else v[:256] + "..."
        elif isinstance(v, dict) and max_depth > 0:
            result[k] = snapshot_sanitize(v, max_depth - 1)
        elif isinstance(v, list) and max_depth > 0:
            result[k] = [
                snapshot_sanitize(i, max_depth - 1) if isinstance(i, dict) else str(i)[:128]
                for i in v[:10]
            ]
        else:
            result[k] = f"<{type(v).__name__}>"
    return result
