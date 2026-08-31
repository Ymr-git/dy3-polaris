"""Dy3+ Polaris 系统入口点 — 启动全栈八层统一应用 (L0-L7 + 前端系统).

该入口点通过 :meth:`UnifiedApp.create_full_app_builder` 组装 L0→L7 全部八层,
并以 Starlette ASGI 应用形式交给 uvicorn 启动。根路径 ``/`` 返回系统前端
控制台页面 (TRAE 风格壳层 + 总览看板 + 学情 + 知识问答)。

命令行参数::

    --host       绑定地址 (默认 0.0.0.0)
    --port       监听端口 (默认 8000)
    --log-level  日志级别 (默认 info)

使用示例::

    python -m dy3_polaris.main --host 0.0.0.0 --port 8000 --log-level info

或作为 ASGI 应用直接启动::

    uvicorn dy3_polaris.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

def _build_app():
    """构建全栈应用 (L0-L7 + 前端系统)."""
    from dy3_polaris.l5.unified_app import UnifiedApp

    builder = UnifiedApp.create_full_app_builder()
    return builder, builder.create_app()


#: 模块级 ASGI 应用 (立即构建, 支持 ``uvicorn dy3_polaris.main:app`` 直接启动).
#: 注意: 若保留为 None 惰性构建, uvicorn 会拿到 None 导致 500 (NoneType not callable).
app = _build_app()[1]


def _ensure_app():
    """惰性构建模块级 app."""
    global app
    if app is None:
        _builder, app = _build_app()  # type: ignore[assignment]
    return app


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器.

    Returns:
        配置好 host/port/log-level 参数的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="Dy3+ Polaris 多智能体协同决策系统",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="绑定地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口 (默认: 8000)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="日志级别 (默认: info), 可选: debug/info/warning/error/critical",
    )
    return parser


def _configure_logging(level: str) -> None:
    """配置根日志记录器.

    Args:
        level: 日志级别字符串 (如 "info"/"INFO"), 不区分大小写。
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _print_banner(layer_count: int, endpoint_count: int, host: str, port: int) -> None:
    """打印启动横幅.

    Args:
        layer_count: 挂载的层数。
        endpoint_count: 端点总数。
        host: 监听地址。
        port: 监听端口。
    """
    bar = "=" * 60
    print(bar)
    print("Dy3+ Polaris 多智能体协同决策系统")
    print(bar)
    print("Dy3+ Polaris 启动中...")
    print(f"  挂载层数: {layer_count} (L0-L7)")
    print(f"  端点总数: {endpoint_count}")
    print(f"  监听地址: http://{host}:{port}")
    print(f"  前端控制台: http://{host}:{port}/")
    print(bar)


def main(argv: Sequence[str] | None = None) -> None:
    """启动 Dy3+ Polaris 全栈八层统一应用.

    解析命令行参数, 配置日志, 组装全部八层 UnifiedApp 并以 uvicorn 启动。

    Args:
        argv: 可选的参数列表, 默认从 ``sys.argv`` 解析 (便于测试)。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("dy3_polaris.main")

    # 延迟导入: 确保参数解析与日志配置先就绪, 且避免模块加载阶段引入重依赖
    from dy3_polaris.l5.unified_app import UnifiedApp

    logger.info("正在组装全栈八层统一应用 (L0-L7)...")
    app_builder = UnifiedApp.create_full_app_builder()
    app_instance = app_builder.create_app()

    # 收集路由摘要用于启动横幅
    routes = app_builder.get_routes_summary()
    layers = sorted({r["layer"] for r in routes})
    _print_banner(
        layer_count=len(layers),
        endpoint_count=len(routes),
        host=args.host,
        port=args.port,
    )
    logger.info("共挂载 %d 层, %d 个端点", len(layers), len(routes))

    import uvicorn

    uvicorn.run(app_instance, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
