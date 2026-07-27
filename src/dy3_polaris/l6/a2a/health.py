"""Agent 健康评分与智能路由.

基于历史任务执行数据为每个 Agent 计算健康评分，
支持按能力选择最优 Agent 的智能路由。

评分维度:
- 成功率 (weight=0.35): 历史任务完成比例
- 延迟 (weight=0.25): 平均/最近延迟
- 稳定性 (weight=0.20): 延迟波动（变异系数）
- 活跃度 (weight=0.10): 近期心跳/任务活跃程度
- 负载 (weight=0.10): 当前并发任务占比

评分范围: [0, 100]，越高越健康。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _AgentStats:
    """单个 Agent 的运行时统计."""
    agent_id: str
    # 任务计数
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    timeout: int = 0
    # 延迟样本（毫秒）
    latencies: list[float] = field(default_factory=list)
    max_latency_samples: int = 50
    # 心跳
    last_heartbeat: float = 0.0
    # 并发
    current_concurrent: int = 0
    max_concurrent: int = 5

    @property
    def total_tasks(self) -> int:
        return self.completed + self.failed + self.cancelled + self.timeout

    @property
    def success_rate(self) -> float:
        t = self.total_tasks
        return self.completed / t if t > 0 else 1.0  # 无数据默认满分

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def latency_cv(self) -> float:
        """延迟变异系数 (coefficient of variation)."""
        if len(self.latencies) < 2:
            return 0.0
        avg = self.avg_latency_ms
        if avg == 0:
            return 0.0
        variance = sum((l - avg) ** 2 for l in self.latencies) / len(self.latencies)
        return math.sqrt(variance) / avg

    @property
    def load_ratio(self) -> float:
        return self.current_concurrent / self.max_concurrent if self.max_concurrent > 0 else 1.0

    def record_result(self, status: str, latency_ms: float = 0.0) -> None:
        if status == "completed":
            self.completed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "cancelled":
            self.cancelled += 1
        elif status == "timeout":
            self.timeout += 1

        if latency_ms > 0 and status == "completed":
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_samples:
                self.latencies = self.latencies[-self.max_latency_samples:]

    def record_heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_tasks": self.total_tasks,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "latency_cv": round(self.latency_cv, 4),
            "load_ratio": round(self.load_ratio, 4),
            "current_concurrent": self.current_concurrent,
            "max_concurrent": self.max_concurrent,
        }


class AgentHealthTracker:
    """Agent 健康评分与智能路由.

    为每个 Agent 维护运行时统计，计算健康评分，
    并支持按能力选择最优 Agent。

    使用示例:
        tracker = AgentHealthTracker()
        tracker.record_task_result("assess-agent", "completed", latency_ms=120)
        score = tracker.health_score("assess-agent")
        best = tracker.select_best_agent(["a1", "a2"], "knowledge_assessment")
    """

    # 评分权重
    W_SUCCESS = 0.35
    W_LATENCY = 0.25
    W_STABILITY = 0.20
    W_ACTIVITY = 0.10
    W_LOAD = 0.10

    def __init__(self) -> None:
        self._stats: dict[str, _AgentStats] = {}

    def _ensure(self, agent_id: str, max_concurrent: int = 5) -> _AgentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = _AgentStats(
                agent_id=agent_id,
                max_concurrent=max_concurrent,
            )
        return self._stats[agent_id]

    # --------------------------------------------------------
    # 事件记录
    # --------------------------------------------------------

    def record_task_result(
        self,
        agent_id: str,
        status: str,
        latency_ms: float = 0.0,
        *,
        max_concurrent: int = 5,
    ) -> None:
        """记录任务执行结果.

        Args:
            agent_id: Agent ID
            status: 任务状态 (completed/failed/cancelled/timeout)
            latency_ms: 执行延迟（毫秒）
            max_concurrent: Agent 的最大并发数
        """
        stats = self._ensure(agent_id, max_concurrent)
        stats.record_result(status, latency_ms)

    def record_heartbeat(self, agent_id: str) -> None:
        """记录心跳."""
        stats = self._ensure(agent_id)
        stats.record_heartbeat()

    def update_concurrency(self, agent_id: str, current: int, max_concurrent: int = 5) -> None:
        """更新并发信息."""
        stats = self._ensure(agent_id, max_concurrent)
        stats.current_concurrent = current
        stats.max_concurrent = max_concurrent

    # --------------------------------------------------------
    # 健康评分
    # --------------------------------------------------------

    def health_score(self, agent_id: str) -> float:
        """计算 Agent 健康评分.

        Returns:
            0-100 的评分
        """
        stats = self._stats.get(agent_id)
        if stats is None:
            return 50.0  # 未知 Agent 中等分

        # 1. 成功率分 [0, 100]
        score_success = stats.success_rate * 100

        # 2. 延迟分 [0, 100] — 200ms 以下满分，2000ms 以上 0 分
        avg_lat = stats.avg_latency_ms
        if avg_lat <= 0:
            score_latency = 100.0  # 无数据默认满分
        elif avg_lat <= 200:
            score_latency = 100.0
        elif avg_lat >= 2000:
            score_latency = 0.0
        else:
            score_latency = 100.0 * (1.0 - (avg_lat - 200) / 1800)

        # 3. 稳定性分 [0, 100] — CV 越小越好
        cv = stats.latency_cv
        if cv <= 0.1:
            score_stability = 100.0
        elif cv >= 1.0:
            score_stability = 0.0
        else:
            score_stability = 100.0 * (1.0 - (cv - 0.1) / 0.9)

        # 4. 活跃度分 [0, 100] — 最近 60s 有心跳满分
        if stats.last_heartbeat > 0:
            age = time.time() - stats.last_heartbeat
            if age <= 60:
                score_activity = 100.0
            elif age >= 300:
                score_activity = 0.0
            else:
                score_activity = 100.0 * (1.0 - (age - 60) / 240)
        else:
            score_activity = 50.0  # 无心跳记录中等分

        # 5. 负载分 [0, 100] — 负载越低越好
        load = stats.load_ratio
        score_load = max(0.0, 100.0 * (1.0 - load))

        # 加权
        score = (
            self.W_SUCCESS * score_success
            + self.W_LATENCY * score_latency
            + self.W_STABILITY * score_stability
            + self.W_ACTIVITY * score_activity
            + self.W_LOAD * score_load
        )

        return round(score, 2)

    def health_report(self, agent_id: str) -> dict[str, Any]:
        """生成详细健康报告."""
        stats = self._stats.get(agent_id)
        return {
            **(stats.to_dict() if stats else {"agent_id": agent_id}),
            "health_score": self.health_score(agent_id),
        }

    def select_best_agent(
        self,
        candidate_ids: list[str],
        capability: str = "",
    ) -> str | None:
        """从候选中选择健康评分最高的 Agent.

        Args:
            candidate_ids: 候选 Agent ID 列表
            capability: 请求的能力名称（用于日志）

        Returns:
            最优 Agent ID，若无候选则返回 None
        """
        if not candidate_ids:
            return None

        best_id = max(candidate_ids, key=lambda aid: self.health_score(aid))
        return best_id

    def all_health_reports(self) -> list[dict[str, Any]]:
        """获取所有 Agent 的健康报告."""
        return [self.health_report(aid) for aid in self._stats]

    def export(self) -> dict[str, Any]:
        """导出汇总."""
        scores = {aid: self.health_score(aid) for aid in self._stats}
        return {
            "total_agents": len(self._stats),
            "health_scores": scores,
            "reports": self.all_health_reports(),
        }

    def reset(self) -> None:
        self._stats.clear()


__all__ = ["AgentHealthTracker"]
