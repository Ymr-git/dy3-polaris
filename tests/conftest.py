"""Pytest 全局配置文件.

职责:
1. 注册 pytest-asyncio marker (兼容旧版配置)
2. 配置 sys.path 确保源码可导入
3. 提供共享 fixture
4. 数据目录隔离: 测试写入临时目录, 根治对生产 l2/data 的污染
"""

import os
import sys
from pathlib import Path

import pytest

# 确保 src/ 在 Python 路径中
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_dir(tmp_path_factory):
    """整个测试会话使用独立数据目录 (DY3_DATA_DIR), 与生产数据完全隔离."""
    data_dir = tmp_path_factory.mktemp("dy3-test-data")
    os.environ["DY3_DATA_DIR"] = str(data_dir)
    # 历史测试显式运行在 demo 模式；产品启动默认不播种这些数据。
    os.environ["DY3_SEED_DEMO_DATA"] = "1"
    # Unit/integration tests never spend external model credits implicitly.
    # Multi-model routing tests opt in with mocked HTTP transports.
    os.environ["DY3_MULTI_MODEL_ENABLED"] = "0"
    # Product tests must exercise the reviewed snapshot, not the legacy
    # hard-coded textbook placeholder.  Individual compatibility tests may
    # still opt in explicitly when they are testing that legacy boundary.
    os.environ.pop("DY3_ENABLE_PLACEHOLDER_KNOWLEDGE", None)
    yield data_dir
    os.environ.pop("DY3_DATA_DIR", None)
    os.environ.pop("DY3_SEED_DEMO_DATA", None)
    os.environ.pop("DY3_MULTI_MODEL_ENABLED", None)


# 注册 asyncio marker (向后兼容)
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an asyncio coroutine (use pytest-asyncio)",
    )
