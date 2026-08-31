"""L2 个性化层异常体系.

继承 L6Error 模式 (参考 L1 层 auth.py 中 L1AuthError / context_broker.py 中 L1ContextError),
所有 L2 异常归属 JSON-RPC -32300 范围, 与各层异常码段隔离:

- L2_ERROR            -> -32300  (L2 基础异常)
- PROFILE_NOT_FOUND   -> -32301  (学情画像未找到)
- TRACING_ERROR       -> -32302  (知识追踪错误)
- IRT_ERROR           -> -32303  (IRT 能力估计错误)
- MEMORY_ERROR        -> -32304  (记忆层错误)
- STORE_ERROR         -> -32305  (存储层错误)

注意: ``MemoryError`` 类名在 l2 模块命名空间内会遮蔽 Python 内置 ``MemoryError``,
这与 l1/auth.py 中异常命名约定一致, 在模块作用域内是安全的
(内置 ``MemoryError`` 仍可通过 ``builtins.MemoryError`` 访问).
"""

from __future__ import annotations

from typing import Any

from dy3_polaris.l6.core.exceptions import L6Error


# ============================================================
# 1. L2 基础异常 (JSON-RPC -32300)
# ============================================================


class L2Error(L6Error):
    """L2 个性化层基础异常 (JSON-RPC -32300).

    所有 L2 层异常的基类, 继承 L6Error, 保留 code/detail/context 三元组结构.

    Attributes:
        code: 机器可读错误码 (默认 "L2_ERROR")
        detail: 人类可读错误详情
        context: 结构化上下文信息, 用于日志和调试
    """

    def __init__(
        self,
        code: str = "L2_ERROR",
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, detail, context)

    def _jsonrpc_code(self) -> int:
        """映射到 JSON-RPC 错误码 (-32300)."""
        return -32300


# ============================================================
# 2. 学情画像异常 (JSON-RPC -32301)
# ============================================================


class ProfileNotFoundError(L2Error):
    """学情画像未找到 (JSON-RPC -32301).

    学习者画像不存在或尚未构建.

    Attributes:
        learner_id: 学习者 ID (自动写入 context)
    """

    def __init__(
        self,
        learner_id: str,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx: dict[str, Any] = {"learner_id": learner_id}
        if context:
            ctx.update(context)
        super().__init__(
            "PROFILE_NOT_FOUND",
            detail or f"学情画像未找到: {learner_id}",
            ctx,
        )

    def _jsonrpc_code(self) -> int:
        return -32301


# ============================================================
# 3. 知识追踪异常 (JSON-RPC -32302)
# ============================================================


class TracingError(L2Error):
    """知识追踪错误 (JSON-RPC -32302).

    BKT 参数更新失败、追踪状态异常等.

    Attributes:
        detail: 人类可读错误详情
        context: 结构化上下文 (如 kp_id)
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("TRACING_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32302


# ============================================================
# 4. IRT 能力估计异常 (JSON-RPC -32303)
# ============================================================


class IRTError(L2Error):
    """IRT 能力估计错误 (JSON-RPC -32303).

    theta 收敛失败、能力估计异常等.

    Attributes:
        detail: 人类可读错误详情
        context: 结构化上下文 (如 learner_id)
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("IRT_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32303


# ============================================================
# 5. 记忆层异常 (JSON-RPC -32304)
# ============================================================


class MemoryError(L2Error):  # noqa: A001  (intentional shadow of builtin in l2 ns)
    """记忆层错误 (JSON-RPC -32304).

    记忆图谱写入/检索失败等.

    .. note::
       本类名在 l2 模块命名空间内遮蔽 Python 内置 ``MemoryError``,
       与 l1/auth.py 等模块的命名约定一致, 模块内安全.
       如需引用内置异常, 请使用 ``builtins.MemoryError``.

    Attributes:
        detail: 人类可读错误详情
        context: 结构化上下文 (如 session_id)
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("MEMORY_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32304


# ============================================================
# 6. 存储层异常 (JSON-RPC -32305)
# ============================================================


class StoreError(L2Error):
    """存储层错误 (JSON-RPC -32305).

    持久化读写失败、存储后端不可用等.

    Attributes:
        detail: 人类可读错误详情
        context: 结构化上下文 (如 operation)
    """

    def __init__(
        self,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("STORE_ERROR", detail, context)

    def _jsonrpc_code(self) -> int:
        return -32305


# ============================================================
# __all__
# ============================================================

__all__ = [
    "L2Error",
    "ProfileNotFoundError",
    "TracingError",
    "IRTError",
    "MemoryError",
    "StoreError",
]
