"""CC4 三横切集成 — 断路器 (Circuit Breaker).

实现 Hystrix/Resilience4j 风格的三态断路器, 保护 CC1/CC2/CC3 模块
免受级联故障影响.

三态模型:
- CLOSED: 正常运行, 请求通过. 连续失败达阈值 → OPEN
- OPEN: 熔断状态, 请求被拒绝. 超时后 → HALF_OPEN
- HALF_OPEN: 探测状态, 限量请求通过. 成功达阈值 → CLOSED, 失败 → OPEN

融合方案:
- Hystrix: 隔舱模式 + 断路器 + 后备策略
- Resilience4j: 函数式断路器 + 事件驱动
- Istio: Service Mesh 级断路器 (Outlier Detection)
- Envoy: 主动健康检查 + 被动断路
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .models import (
    CircuitBreakerConfig,
    CircuitState,
)
from .exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """断路器 — 保护模块免受级联故障.

    使用示例::

        breaker = CircuitBreaker("cc1")
        try:
            result = breaker.call(lambda: cc1_pipeline.review(request))
        except CircuitBreakerOpenError:
            # 降级处理
            pass

    状态转换::

        CLOSED ──(连续失败≥阈值)──→ OPEN
           ↑                          │
           │                          │ (超时后)
           │                          ↓
           └──(成功≥阈值)──────── HALF_OPEN
                                       │
                                       │ (失败)
                                       ↓
                                      OPEN
    """

    def __init__(
        self,
        module: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """初始化断路器.

        Args:
            module: 保护的模块名 (cc1/cc2/cc3)
            config: 断路器配置, 为 None 时使用默认配置
        """
        self._module = module
        self._config = config or CircuitBreakerConfig(module=module)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._half_open_calls = 0
        self._total_trips = 0
        self._event_log: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------

    @property
    def module(self) -> str:
        return self._module

    @property
    def state(self) -> CircuitState:
        """当前断路器状态 (含自动恢复检查)."""
        self._check_recovery()
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def total_trips(self) -> int:
        return self._total_trips

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    # --------------------------------------------------------
    # 核心方法
    # --------------------------------------------------------

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """通过断路器执行调用.

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            CircuitBreakerOpenError: 断路器开启时
            Exception: 函数本身的异常 (同时记录失败)
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            retry_after = max(
                0.0,
                self._config.recovery_timeout
                - (time.time() - self._opened_at),
            )
            logger.warning(
                "断路器开启: 模块=%s, %0.1fs 后恢复",
                self._module,
                retry_after,
            )
            raise CircuitBreakerOpenError(self._module, retry_after)

        if (
            current_state == CircuitState.HALF_OPEN
            and self._half_open_calls >= self._config.half_open_max_calls
        ):
            raise CircuitBreakerOpenError(self._module, 0.0)

        # 执行调用
        if current_state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """记录成功."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        """记录失败."""
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _check_recovery(self) -> None:
        """检查是否应从 OPEN 转为 HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return

        elapsed = time.time() - self._opened_at
        if elapsed >= self._config.recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换."""
        old_state = self._state
        self._state = new_state

        event = {
            "timestamp": time.time(),
            "module": self._module,
            "from": old_state.value,
            "to": new_state.value,
        }
        self._event_log.append(event)

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._total_trips += 1
            logger.warning(
                "断路器跳闸: 模块=%s, %s → %s (连续失败=%d)",
                self._module,
                old_state.value,
                new_state.value,
                self._failure_count,
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
            logger.info(
                "断路器半开: 模块=%s, 开始探测",
                self._module,
            )
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(
                "断路器关闭: 模块=%s, 恢复正常",
                self._module,
            )

    # --------------------------------------------------------
    # 状态查询
    # --------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """获取断路器状态摘要."""
        current = self.state
        return {
            "module": self._module,
            "state": current.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_trips": self._total_trips,
            "failure_threshold": self._config.failure_threshold,
            "recovery_timeout_s": self._config.recovery_timeout,
            "last_failure_time": self._last_failure_time,
            "opened_at": self._opened_at,
        }

    def get_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近的状态转换事件."""
        return list(reversed(self._event_log[-limit:]))

    def reset(self) -> None:
        """重置断路器到 CLOSED 状态."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info("断路器手动重置: 模块=%s", self._module)


__all__ = ["CircuitBreaker"]
