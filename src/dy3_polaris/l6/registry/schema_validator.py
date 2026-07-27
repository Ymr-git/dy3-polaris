"""JSON Schema 校验器.

提供 Dy3+ Polaris L6 工具注册中心使用的 Schema 校验能力：
- 工具定义 Schema 合法性校验（Draft 2020-12）
- 工具调用参数校验（input_schema）
- 工具输出结构校验（output_schema）
- 细粒度错误路径报告

依赖 jsonschema 库实现核心校验逻辑。
"""

from __future__ import annotations

import logging
from typing import Any

from jsonschema import Draft202012Validator

from ..core.exceptions import InvalidParamsError, SchemaValidationError

logger = logging.getLogger(__name__)


# ============================================================
# JSON Schema 基础模板
# ============================================================

INPUT_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

OUTPUT_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "data": {"type": "object"},
    },
    "required": ["success"],
}


# ============================================================
# Schema 校验器
# ============================================================

class SchemaValidator:
    """JSON Schema 校验器.

    支持两种校验模式：
    1. 结构校验：验证工具的 input_schema / output_schema 本身是否合法
    2. 值校验：验证实际参数/输出是否符合对应 Schema

    使用示例:
        validator = SchemaValidator()
        validator.validate_definition(tool_registration)
        validator.validate_input(tool_registration, {"learner_id": "u001"})
    """

    def __init__(self) -> None:
        self._cache: dict[str, Draft202012Validator] = {}

    # ---- Schema 定义合法性校验 ----

    def validate_input_schema(self, schema: dict[str, Any]) -> list[SchemaValidationError]:
        """校验 input_schema 结构是否合法.

        Returns:
            错误列表，空列表表示通过。
        """
        errors: list[SchemaValidationError] = []

        if not isinstance(schema, dict):
            errors.append(SchemaValidationError("$", "input_schema must be a dict"))
            return errors

        if schema.get("type") != "object":
            errors.append(
                SchemaValidationError("$.type", f"input_schema type must be 'object', got '{schema.get('type')}'")
            )

        if "properties" not in schema:
            errors.append(SchemaValidationError("$.properties", "input_schema must have 'properties' field"))

        if "required" not in schema:
            errors.append(SchemaValidationError("$.required", "input_schema must have 'required' field"))

        # 使用 Draft202012Validator 校验 schema 本身的合法性
        try:
            meta_schema = Draft202012Validator.META_SCHEMA
            meta_validator = Draft202012Validator(meta_schema)
            for err in meta_validator.iter_errors(schema):
                path = "$." + ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "$"
                errors.append(SchemaValidationError(path, err.message, err.instance))
        except Exception as exc:
            errors.append(SchemaValidationError("$", f"Schema meta-validation failed: {exc}"))

        return errors

    def validate_output_schema(self, schema: dict[str, Any] | None) -> list[SchemaValidationError]:
        """校验 output_schema 结构是否合法.

        output_schema 为 None 时不校验（允许工具不声明输出 Schema）。
        """
        if schema is None:
            return []

        errors: list[SchemaValidationError] = []

        if not isinstance(schema, dict):
            errors.append(SchemaValidationError("$", "output_schema must be a dict or None"))
            return errors

        try:
            meta_schema = Draft202012Validator.META_SCHEMA
            meta_validator = Draft202012Validator(meta_schema)
            for err in meta_validator.iter_errors(schema):
                path = "$." + ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "$"
                errors.append(SchemaValidationError(path, err.message, err.instance))
        except Exception as exc:
            errors.append(SchemaValidationError("$", f"Schema meta-validation failed: {exc}"))

        return errors

    def validate_definition(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
    ) -> list[SchemaValidationError]:
        """校验工具定义的完整合法性.

        检查：
        - name 符合命名规范
        - description 非空
        - input_schema 结构合法
        - output_schema 结构合法（如有）
        """
        errors: list[SchemaValidationError] = []

        # name 校验
        import re
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            errors.append(
                SchemaValidationError(
                    "name",
                    f"Tool name '{name}' must match pattern: ^[a-z][a-z0-9_]*$",
                )
            )

        # description 校验
        if not description or not description.strip():
            errors.append(SchemaValidationError("description", "description must be non-empty"))

        # input_schema 校验
        errors.extend(self.validate_input_schema(input_schema))

        # output_schema 校验
        errors.extend(self.validate_output_schema(output_schema))

        return errors

    # ---- 值校验 ----

    def validate_input(
        self,
        input_schema: dict[str, Any],
        arguments: dict[str, Any],
        *,
        tool_name: str = "",
    ) -> None:
        """校验工具调用参数.

        Raises:
            InvalidParamsError: 参数不符合 input_schema 时
        """
        validator = self._get_or_create_validator(input_schema, "input")

        errors = list(validator.iter_errors(arguments))
        if errors:
            # 收集所有错误，提供详细路径
            detail_parts: list[str] = []
            for err in errors:
                path = ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "(root)"
                detail_parts.append(f"{path}: {err.message}")

            detail = "; ".join(detail_parts[:5])  # 最多报告前 5 个错误
            if len(errors) > 5:
                detail += f" (and {len(errors) - 5} more)"

            raise InvalidParamsError(
                detail=f"[{tool_name}] {detail}" if tool_name else detail,
                context={"error_count": len(errors), "tool_name": tool_name},
            )

    def validate_output(
        self,
        output_schema: dict[str, Any] | None,
        result: Any,
        *,
        tool_name: str = "",
    ) -> None:
        """校验工具输出.

        output_schema 为 None 时跳过校验。
        """
        if output_schema is None:
            return

        validator = self._get_or_create_validator(output_schema, "output")

        errors = list(validator.iter_errors(result))
        if errors:
            detail_parts: list[str] = []
            for err in errors:
                path = ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "(root)"
                detail_parts.append(f"{path}: {err.message}")

            detail = "; ".join(detail_parts[:5])
            if len(errors) > 5:
                detail += f" (and {len(errors) - 5} more)"

            raise SchemaValidationError(
                path=f"output.{tool_name}" if tool_name else "output",
                message=detail,
                context={"error_count": len(errors), "tool_name": tool_name},
            )

    # ---- 内部方法 ----

    def _get_or_create_validator(self, schema: dict[str, Any], cache_key_prefix: str) -> Draft202012Validator:
        """获取或创建缓存的 Validator."""
        import hashlib
        import json

        schema_str = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        cache_key = f"{cache_key_prefix}:{schema_hash}"

        if cache_key not in self._cache:
            self._cache[cache_key] = Draft202012Validator(schema)

        return self._cache[cache_key]

    def clear_cache(self) -> None:
        """清空 validator 缓存."""
        self._cache.clear()


# ============================================================
# 全局单例
# ============================================================

_global_validator: SchemaValidator | None = None


def get_validator() -> SchemaValidator:
    """获取全局 SchemaValidator 单例."""
    global _global_validator
    if _global_validator is None:
        _global_validator = SchemaValidator()
    return _global_validator


def reset_validator() -> None:
    """重置全局 validator（仅用于测试）."""
    global _global_validator
    _global_validator = None
