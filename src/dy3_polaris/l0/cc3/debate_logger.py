"""CC3 溯源捕获层 — 辩论日志引擎 (Debate Logger).

实现三级辩论日志系统, 完整记录辩论生命周期:
- Pre-Debate: 触发原因、边界预设、参与 Agent 配置
- During-Debate: 多轮辩论 (Generator 论点 ↔ Reviewer 反驳)
- Post-Debate: 裁决结果、资源消耗、辩论结果影响

三级存储策略:
- Summary (永久): 仅元数据 + 收敛结果 + 裁决 → L0 常规表
- Full (90天→冷存储): + 全部轮次 + 资源消耗 → L0 Archive 表
- Debug (30天清理): + Prompt 输入输出 (脱敏) → Session Fork/对象存储

核心能力:
- 辩论日志创建、轮次追加、裁决记录、结果更新
- 分歧度曲线跟踪与收敛判定
- 不可变哈希链校验 (tamper-evident)
- Prompt 脱敏处理 (API key/PII 清除)
- 5 秒内持久化承诺
- 辩论日志查询与统计

融合方案:
- RFC 6962 Certificate Transparency: append-only 日志 + Merkle 证明
- W3C PROV: Activity (辩论过程) + Agent (Generator/Reviewer/Adjudicator)
- OpenTelemetry GenAI: trace_id/span_id 标准化传递
- Langfuse: LLM trace 树可视化 + Prompt 脱敏
- GAIA: 三阶段协商协议 (Screening → Negotiation → Execution)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from typing import Any

from .models import (
    AdjudicatorVerdict,
    ConvergenceStatus,
    CounterType,
    DebateArgument,
    DebateCounterargument,
    DebateLog,
    DebateOutcome,
    DebateResourceUsage,
    DebateRound,
    LogVerbosity,
    PreDebateRecord,
    SourceTier,
)
from .exceptions import (
    CC3Error,
    DebateLogNotFoundError,
    HashMismatchError,
)

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 脱敏处理器
# ============================================================


class PromptSanitizer:
    """Prompt 脱敏处理器.

    在 Debug 级别日志中, 对 Prompt 进行脱敏处理:
    - API Key / Secret / Token 清除
    - 个人邮箱/手机号/身份证号掩码
    - 敏感路径 (/home/user/xxx) 清除
    - IP 地址保留前两段

    融合方案:
    - Langfuse: Prompt 脱敏最佳实践
    - OWASP: 敏感数据最小化原则
    """

    # 敏感模式正则
    _API_KEY_PATTERNS = [
        re.compile(r"(?:api[_-]?key|api[_-]?secret|access[_-]?token|bearer)\s*[:=]\s*\S+", re.IGNORECASE),
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI API key
        re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    ]
    _EMAIL_PATTERN = re.compile(r"\b([a-zA-Z0-9._%+-])[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")
    _PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")
    _ID_CARD_PATTERN = re.compile(r"\b\d{17}[\dXx]\b")
    _IP_PATTERN = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
    _PATH_PATTERN = re.compile(r"/(?:home|Users|root|var|opt|etc)/\S+")

    @classmethod
    def sanitize(cls, text: str) -> str:
        """对文本进行脱敏处理.

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not text:
            return text

        result = text

        # API Key / Secret / Token
        for pattern in cls._API_KEY_PATTERNS:
            result = pattern.sub("[REDACTED]", result)

        # 邮箱 (保留首字母 + 域名)
        result = cls._EMAIL_PATTERN.sub(
            lambda m: f"{m.group(1)}***@{m.group(2)}", result
        )

        # 手机号 (保留前3后4)
        result = cls._PHONE_PATTERN.sub(
            lambda m: m.group()[:3] + "****" + m.group()[-4:], result
        )

        # 身份证号 (保留前6后4)
        result = cls._ID_CARD_PATTERN.sub(
            lambda m: m.group()[:6] + "********" + m.group()[-4:], result
        )

        # IP 地址 (保留前两段)
        result = cls._IP_PATTERN.sub(
            lambda m: f"{m.group(1)}.{m.group(2)}.x.x", result
        )

        # 敏感路径
        result = cls._PATH_PATTERN.sub("[REDACTED_PATH]", result)

        return result

    @classmethod
    def sanitize_prompt_record(
        cls,
        role: str,
        prompt_input: str,
        prompt_output: str,
        model_name: str = "",
        round_number: int = 0,
    ) -> dict[str, Any]:
        """创建脱敏后的 Prompt 记录.

        Args:
            role: Agent 角色 (generator/reviewer/adjudicator)
            prompt_input: 输入 Prompt
            prompt_output: 输出内容
            model_name: 模型名称
            round_number: 轮次

        Returns:
            脱敏后的 Prompt 记录字典
        """
        return {
            "record_id": f"prompt-{uuid.uuid4().hex[:8]}",
            "role": role,
            "round_number": round_number,
            "model_name": model_name,
            "input_sanitized": cls.sanitize(prompt_input),
            "output_sanitized": cls.sanitize(prompt_output),
            "timestamp": time.time(),
        }


# ============================================================
# 分歧度计算器
# ============================================================


class DivergenceCalculator:
    """分歧度计算器.

    计算 Generator 与 Reviewer 之间的立场分歧度 (0.0-1.0):
    - 0.0: 完全一致
    - 1.0: 完全对立

    方法:
    - argument_based: 基于论点/反驳的覆盖度计算
    - embedding_based: 基于词频向量的余弦距离 (简化版)
    - confidence_gap: 基于置信度差异
    """

    @staticmethod
    def calculate(
        generator_args: list[DebateArgument],
        reviewer_counters: list[DebateCounterargument],
        method: str = "argument_based",
    ) -> float:
        """计算单轮分歧度.

        Args:
            generator_args: Generator 论点列表
            reviewer_counters: Reviewer 反驳列表
            method: 计算方法

        Returns:
            分歧度 (0.0-1.0)
        """
        if not generator_args and not reviewer_counters:
            return 0.0
        if not generator_args or not reviewer_counters:
            # 单方发言, 分歧度中等偏高
            return 0.5

        if method == "argument_based":
            return DivergenceCalculator._argument_based(
                generator_args, reviewer_counters
            )
        elif method == "confidence_gap":
            return DivergenceCalculator._confidence_gap(
                generator_args, reviewer_counters
            )
        else:
            return DivergenceCalculator._argument_based(
                generator_args, reviewer_counters
            )

    @staticmethod
    def _argument_based(
        generator_args: list[DebateArgument],
        reviewer_counters: list[DebateCounterargument],
    ) -> float:
        """基于论点覆盖度的分歧度计算.

        被反驳的论点比例越高, 分歧度越大。
        同时考虑反驳的置信度与论点置信度差异。
        """
        total_args = len(generator_args)
        targeted_args: set[str] = set()
        confidence_diff_sum = 0.0

        for counter in reviewer_counters:
            for target_id in counter.targets:
                targeted_args.add(target_id)
            # 置信度差异
            for arg in generator_args:
                if arg.point_id in counter.targets:
                    confidence_diff_sum += abs(arg.confidence - counter.confidence)

        # 被反驳比例
        targeted_ratio = len(targeted_args) / total_args if total_args > 0 else 0.0

        # 平均置信度差异
        avg_conf_diff = (
            confidence_diff_sum / len(targeted_args)
            if targeted_args
            else 0.0
        )

        # 综合分歧度: 0.6 * 被反驳比例 + 0.4 * 置信度差异
        divergence = 0.6 * targeted_ratio + 0.4 * min(avg_conf_diff, 1.0)

        return round(min(max(divergence, 0.0), 1.0), 4)

    @staticmethod
    def _confidence_gap(
        generator_args: list[DebateArgument],
        reviewer_counters: list[DebateCounterargument],
    ) -> float:
        """基于置信度差异的分歧度计算."""
        if not generator_args:
            return 0.0

        gen_avg_conf = sum(a.confidence for a in generator_args) / len(generator_args)
        rev_avg_conf = (
            sum(c.confidence for c in reviewer_counters) / len(reviewer_counters)
            if reviewer_counters
            else gen_avg_conf
        )

        return round(abs(gen_avg_conf - rev_avg_conf), 4)

    @staticmethod
    def check_convergence(
        divergence_curve: list[float],
        threshold: float = 0.1,
    ) -> tuple[bool, int]:
        """检查是否达到收敛.

        Args:
            divergence_curve: 分歧度曲线 (每轮一个值)
            threshold: 收敛阈值

        Returns:
            (是否收敛, 收敛轮次)
        """
        if not divergence_curve:
            return False, 0

        for i, div in enumerate(divergence_curve):
            if div < threshold:
                return True, i + 1

        return False, 0

    @staticmethod
    def convergence_trend(divergence_curve: list[float]) -> str:
        """分析收敛趋势.

        Returns:
            "converging" (下降), "diverging" (上升), "stable" (稳定), "unknown"
        """
        if len(divergence_curve) < 2:
            return "unknown"

        first_half = divergence_curve[: len(divergence_curve) // 2 + 1]
        second_half = divergence_curve[len(divergence_curve) // 2 :]

        avg_first = sum(first_half) / len(first_half) if first_half else 0.0
        avg_second = sum(second_half) / len(second_half) if second_half else 0.0

        diff = avg_second - avg_first
        if diff < -0.02:
            return "converging"
        elif diff > 0.02:
            return "diverging"
        else:
            return "stable"


# ============================================================
# 辩论日志引擎
# ============================================================


class DebateLogger:
    """辩论日志引擎 — 三级日志系统.

    管理辩论日志的完整生命周期:
    1. create_log(): 创建日志, 记录 Pre-Debate 信息
    2. add_round(): 追加辩论轮次 (During-Debate)
    3. check_convergence(): 检查收敛
    4. record_adjudication(): 记录裁决 (Post-Debate)
    5. record_outcome(): 记录辩论结果影响
    6. finalize(): 完成日志, 计算最终哈希, 标记持久化
    7. verify_integrity(): 验证日志完整性

    三级日志控制:
    - Summary: 仅保留元数据 + 收敛 + 裁决
    - Full: + 轮次 + 资源消耗
    - Debug: + 脱敏 Prompt

    使用示例::

        dl = DebateLogger()
        log = dl.create_log(
            debate_id="debate-001",
            task_id="task-abc",
            complexity_score=45.0,
            verbosity=LogVerbosity.FULL,
        )
        dl.add_round(log.debate_log_id, generator_args, reviewer_counters)
        dl.check_convergence(log.debate_log_id)
        dl.record_adjudication(log.debate_log_id, verdict)
        dl.finalize(log.debate_log_id)
    """

    def __init__(
        self,
        default_verbosity: LogVerbosity = LogVerbosity.SUMMARY,
        default_convergence_threshold: float = 0.1,
        default_max_rounds: int = 3,
        persistence_delay_ms: float = 5000.0,
    ) -> None:
        """初始化辩论日志引擎.

        Args:
            default_verbosity: 默认日志级别
            default_convergence_threshold: 默认收敛阈值
            default_max_rounds: 默认最大轮次
            persistence_delay_ms: 持久化延迟 (毫秒, 设计要求5秒内)
        """
        self._logs: dict[str, DebateLog] = {}
        self._debate_index: dict[str, str] = {}  # debate_id -> debate_log_id
        self._task_index: dict[str, list[str]] = {}  # task_id -> [debate_log_id]
        self._default_verbosity = default_verbosity
        self._default_threshold = default_convergence_threshold
        self._default_max_rounds = default_max_rounds
        self._persistence_delay_ms = persistence_delay_ms
        self._lock = threading.RLock()
        self._divergence_calc = DivergenceCalculator()
        self._sanitizer = PromptSanitizer()

    # ==========================================================
    # 日志创建 (Pre-Debate)
    # ==========================================================

    def create_log(
        self,
        debate_id: str = "",
        task_id: str = "",
        session_id: str = "",
        trigger_reason: str = "",
        complexity_score: float = 0.0,
        threshold_range: str = "31-65",
        focus_area: str = "",
        excluded_topics: list[str] | None = None,
        source_tier_requirement: SourceTier = SourceTier.TIER_2,
        acceptable_evidence_types: list[str] | None = None,
        time_range_constraint: float = 0.0,
        participant_configs: list[dict[str, Any]] | None = None,
        verbosity: LogVerbosity | None = None,
        convergence_threshold: float | None = None,
        max_rounds: int | None = None,
    ) -> DebateLog:
        """创建辩论日志, 记录 Pre-Debate 信息.

        Args:
            debate_id: 辩论 ID
            task_id: 关联任务 ID
            session_id: 会话 ID
            trigger_reason: 触发原因
            complexity_score: 复杂度评分 (31-65 区间触发辩论)
            threshold_range: 触发阈值范围
            focus_area: 辩论焦点
            excluded_topics: 排除的话题
            source_tier_requirement: 来源等级要求
            acceptable_evidence_types: 可接受的证据类型
            time_range_constraint: 时间约束 (秒)
            participant_configs: 参与 Agent 配置
            verbosity: 日志级别 (None=使用默认)
            convergence_threshold: 收敛阈值 (None=使用默认)
            max_rounds: 最大轮次 (None=使用默认)

        Returns:
            新创建的 DebateLog
        """
        with self._lock:
            pre_debate = PreDebateRecord(
                complexity_score=complexity_score,
                threshold_range=threshold_range,
                focus_area=focus_area,
                excluded_topics=excluded_topics or [],
                source_tier_requirement=source_tier_requirement,
                acceptable_evidence_types=acceptable_evidence_types or [],
                time_range_constraint=time_range_constraint,
                participant_configs=participant_configs or [],
            )

            log = DebateLog(
                debate_id=debate_id or f"debate-{uuid.uuid4().hex[:12]}",
                task_id=task_id,
                session_id=session_id,
                trigger_reason=trigger_reason,
                verbosity=verbosity or self._default_verbosity,
                pre_debate=pre_debate,
                convergence_threshold=convergence_threshold or self._default_threshold,
                max_rounds=max_rounds or self._default_max_rounds,
            )

            self._logs[log.debate_log_id] = log
            if debate_id:
                self._debate_index[debate_id] = log.debate_log_id
            if task_id:
                self._task_index.setdefault(task_id, []).append(log.debate_log_id)

            logger.info(
                "创建辩论日志: id=%s, debate=%s, task=%s, verbosity=%s",
                log.debate_log_id,
                log.debate_id,
                task_id,
                log.verbosity.value,
            )
            return log

    # ==========================================================
    # 轮次追加 (During-Debate)
    # ==========================================================

    def add_round(
        self,
        debate_log_id: str,
        generator_arguments: list[DebateArgument] | None = None,
        reviewer_counterarguments: list[DebateCounterargument] | None = None,
        round_duration_ms: float = 0.0,
        divergence: float | None = None,
        divergence_method: str = "argument_based",
        debug_prompts: list[dict[str, Any]] | None = None,
    ) -> DebateRound:
        """追加一个辩论轮次.

        自动计算分歧度 (如果未提供), 更新分歧度曲线, 检查收敛。

        Args:
            debate_log_id: 日志 ID
            generator_arguments: Generator 论点列表
            reviewer_counterarguments: Reviewer 反驳列表
            round_duration_ms: 本轮耗时 (毫秒)
            divergence: 手动指定的分歧度 (None=自动计算)
            divergence_method: 分歧度计算方法
            debug_prompts: 调试级 Prompt 记录 (仅 DEBUG 级别有效)

        Returns:
            创建的 DebateRound

        Raises:
            DebateLogNotFoundError: 日志不存在
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            gen_args = generator_arguments or []
            rev_counters = reviewer_counterarguments or []

            # 计算分歧度
            if divergence is None:
                divergence = self._divergence_calc.calculate(
                    gen_args, rev_counters, divergence_method
                )

            round_number = len(log.rounds) + 1
            round_record = DebateRound(
                round_number=round_number,
                generator_arguments=gen_args,
                reviewer_counterarguments=rev_counters,
                round_divergence=divergence,
                round_timestamp=time.time(),
                round_duration_ms=round_duration_ms,
            )

            log.rounds.append(round_record)
            log.divergence_curve.append(divergence)
            log.final_divergence = divergence

            # 检查收敛
            converged, conv_round = self._divergence_calc.check_convergence(
                log.divergence_curve, log.convergence_threshold
            )
            if converged:
                log.convergence_reached = True
                log.convergence_round = conv_round
                log.convergence_status = ConvergenceStatus.CONVERGED

            # 达到最大轮次
            if round_number >= log.max_rounds and not log.convergence_reached:
                log.convergence_status = ConvergenceStatus.FORCE_RESOLVED

            # Debug 级别: 追加脱敏 Prompt
            if log.verbosity == LogVerbosity.DEBUG and debug_prompts:
                for dp in debug_prompts:
                    sanitized = self._sanitizer.sanitize_prompt_record(
                        role=dp.get("role", ""),
                        prompt_input=dp.get("input", ""),
                        prompt_output=dp.get("output", ""),
                        model_name=dp.get("model_name", ""),
                        round_number=round_number,
                    )
                    log.debug_prompts.append(sanitized)

            # 重算哈希
            log.immutable_hash = log.compute_hash()

            logger.debug(
                "追加辩论轮次: log=%s, round=%d, divergence=%.4f, converged=%s",
                debate_log_id,
                round_number,
                divergence,
                log.convergence_reached,
            )
            return round_record

    # ==========================================================
    # 收敛检查
    # ==========================================================

    def check_convergence(self, debate_log_id: str) -> dict[str, Any]:
        """检查辩论收敛状态.

        Args:
            debate_log_id: 日志 ID

        Returns:
            收敛报告::

                {
                    "converged": bool,
                    "convergence_round": int,
                    "final_divergence": float,
                    "threshold": float,
                    "trend": str,
                    "curve": [float, ...],
                }
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            trend = self._divergence_calc.convergence_trend(log.divergence_curve)

            return {
                "converged": log.convergence_reached,
                "convergence_status": log.convergence_status.value,
                "convergence_round": log.convergence_round,
                "final_divergence": log.final_divergence,
                "threshold": log.convergence_threshold,
                "trend": trend,
                "curve": list(log.divergence_curve),
                "max_rounds": log.max_rounds,
                "current_rounds": len(log.rounds),
            }

    # ==========================================================
    # 裁决记录 (Post-Debate)
    # ==========================================================

    def record_adjudication(
        self,
        debate_log_id: str,
        adjudicator_id: str = "",
        consensus_position: str = "",
        three_dimensional_score: dict[str, float] | None = None,
        adopted_arguments: list[str] | None = None,
        rejected_arguments: list[str] | None = None,
        modification_notes: str = "",
        invocation_reason: str = "",
    ) -> AdjudicatorVerdict:
        """记录裁决结果.

        裁决在以下情况触发:
        - convergence_reached_before_max_rounds: 收敛达成
        - max_rounds_exhausted: 最大轮次耗尽
        - timeout: 超时
        - manual_trigger: 手动触发

        Args:
            debate_log_id: 日志 ID
            adjudicator_id: 裁决 Agent ID
            consensus_position: 共识立场
            three_dimensional_score: 三维评分 {accuracy, completeness, pedagogical_fit}
            adopted_arguments: 采纳的论点 ID 列表
            rejected_arguments: 驳回的论点 ID 列表
            modification_notes: 修改说明
            invocation_reason: 裁决触发原因

        Returns:
            创建的 AdjudicatorVerdict
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            verdict = AdjudicatorVerdict(
                adjudicator_id=adjudicator_id,
                consensus_position=consensus_position,
                three_dimensional_score=three_dimensional_score or {},
                adopted_arguments=adopted_arguments or [],
                rejected_arguments=rejected_arguments or [],
                modification_notes=modification_notes,
            )
            log.adjudicator_verdict = verdict
            log.immutable_hash = log.compute_hash()

            logger.info(
                "记录裁决: log=%s, adjudicator=%s, reason=%s",
                debate_log_id,
                adjudicator_id,
                invocation_reason,
            )
            return verdict

    # ==========================================================
    # 结果记录 (Post-Debate)
    # ==========================================================

    def record_outcome(
        self,
        debate_log_id: str,
        final_consensus: str = "",
        affected_kp_ids: list[str] | None = None,
        kg_relations_updated: list[dict[str, Any]] | None = None,
        bkt_adjustments: list[dict[str, Any]] | None = None,
        adopted_into_kb: bool = False,
        kb_version_after: str = "",
    ) -> DebateOutcome:
        """记录辩论结果影响.

        Args:
            debate_log_id: 日志 ID
            final_consensus: 最终共识
            affected_kp_ids: 受影响的知识点 ID 列表
            kg_relations_updated: 知识图谱关系更新
            bkt_adjustments: BKT 参数调整
            adopted_into_kb: 是否采纳进知识库
            kb_version_after: 知识库版本 (采纳后)

        Returns:
            更新后的 DebateOutcome
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            log.outcome = DebateOutcome(
                final_consensus=final_consensus,
                affected_kp_ids=affected_kp_ids or [],
                kg_relations_updated=kg_relations_updated or [],
                bkt_adjustments=bkt_adjustments or [],
                adopted_into_kb=adopted_into_kb,
                kb_version_after=kb_version_after,
            )
            log.immutable_hash = log.compute_hash()

            logger.info(
                "记录辩论结果: log=%s, adopted=%s, affected_kps=%d",
                debate_log_id,
                adopted_into_kb,
                len(affected_kp_ids or []),
            )
            return log.outcome

    # ==========================================================
    # 资源消耗记录
    # ==========================================================

    def record_resource_usage(
        self,
        debate_log_id: str,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        api_calls: int = 0,
        compute_time_ms: float = 0.0,
        external_tool_calls: int = 0,
        estimated_cost: float = 0.0,
    ) -> DebateResourceUsage:
        """记录辩论资源消耗.

        Args:
            debate_log_id: 日志 ID
            total_tokens: 总 Token 数
            prompt_tokens: Prompt Token 数
            completion_tokens: Completion Token 数
            api_calls: API 调用次数
            compute_time_ms: 计算耗时 (毫秒)
            external_tool_calls: 外部工具调用次数
            estimated_cost: 估算成本 (美元)

        Returns:
            更新后的 DebateResourceUsage
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            log.resource_usage = DebateResourceUsage(
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                api_calls=api_calls,
                compute_time_ms=compute_time_ms,
                external_tool_calls=external_tool_calls,
                estimated_cost=estimated_cost,
            )
            log.immutable_hash = log.compute_hash()
            return log.resource_usage

    # ==========================================================
    # 完成与持久化
    # ==========================================================

    def finalize(self, debate_log_id: str) -> DebateLog:
        """完成辩论日志, 标记持久化时间.

        设计要求: 辩论结束后 5 秒内完成持久化。
        此方法标记 persisted_at 时间戳, 实际持久化由上层 L0 Ledger 处理。

        Args:
            debate_log_id: 日志 ID

        Returns:
            完成的 DebateLog

        Raises:
            DebateLogNotFoundError: 日志不存在
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            # 如果未记录裁决, 标记为中止
            if log.adjudicator_verdict is None and not log.convergence_reached:
                log.convergence_status = ConvergenceStatus.ABORTED

            log.persisted_at = time.time()
            log.immutable_hash = log.compute_hash()

            logger.info(
                "完成辩论日志: id=%s, status=%s, rounds=%d, converged=%s, persisted_at=%.3f",
                debate_log_id,
                log.convergence_status.value,
                len(log.rounds),
                log.convergence_reached,
                log.persisted_at,
            )
            return log

    # ==========================================================
    # 完整性验证
    # ==========================================================

    def verify_integrity(self, debate_log_id: str) -> dict[str, Any]:
        """验证辩论日志的完整性.

        检查:
        - 不可变哈希是否匹配
        - 分歧度曲线与轮次数量是否一致
        - 收敛状态是否正确

        Args:
            debate_log_id: 日志 ID

        Returns:
            验证报告::

                {
                    "debate_log_id": str,
                    "hash_verified": bool,
                    "curve_consistent": bool,
                    "convergence_consistent": bool,
                    "all_passed": bool,
                    "issues": [...],
                }

        Raises:
            HashMismatchError: 哈希不匹配 (可能被篡改)
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            issues: list[str] = []

            # 哈希验证
            hash_verified = log.verify_hash()
            if not hash_verified:
                raise HashMismatchError(
                    expected_hash=log.immutable_hash,
                    actual_hash=log.compute_hash(),
                    record_id=debate_log_id,
                )

            # 分歧度曲线一致性
            curve_consistent = len(log.divergence_curve) == len(log.rounds)
            if not curve_consistent:
                issues.append(
                    f"分歧度曲线长度({len(log.divergence_curve)}) != 轮次数({len(log.rounds)})"
                )

            # 收敛状态一致性
            convergence_consistent = True
            if log.convergence_reached:
                converged, conv_round = self._divergence_calc.check_convergence(
                    log.divergence_curve, log.convergence_threshold
                )
                if not converged:
                    convergence_consistent = False
                    issues.append("标记为收敛但分歧度曲线未达到阈值")
                elif conv_round != log.convergence_round:
                    convergence_consistent = False
                    issues.append(
                        f"收敛轮次不匹配: 标记={log.convergence_round}, 实际={conv_round}"
                    )

            all_passed = hash_verified and curve_consistent and convergence_consistent

            return {
                "debate_log_id": debate_log_id,
                "hash_verified": hash_verified,
                "curve_consistent": curve_consistent,
                "convergence_consistent": convergence_consistent,
                "all_passed": all_passed,
                "issues": issues,
            }

    # ==========================================================
    # 日志导出 (按级别)
    # ==========================================================

    def export_log(
        self,
        debate_log_id: str,
        verbosity: LogVerbosity | None = None,
    ) -> dict[str, Any]:
        """按指定级别导出辩论日志.

        不同级别包含不同详细程度:
        - Summary: 元数据 + 收敛 + 裁决
        - Full: + 全部轮次 + 资源消耗
        - Debug: + 脱敏 Prompt

        Args:
            debate_log_id: 日志 ID
            verbosity: 导出级别 (None=使用日志本身的级别)

        Returns:
            按级别裁剪后的日志字典
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)

            level = verbosity or log.verbosity

            # Summary 级别: 基础信息
            result: dict[str, Any] = {
                "debate_log_id": log.debate_log_id,
                "debate_id": log.debate_id,
                "task_id": log.task_id,
                "session_id": log.session_id,
                "trigger_reason": log.trigger_reason,
                "verbosity": level.value,
                "convergence_status": log.convergence_status.value,
                "convergence_reached": log.convergence_reached,
                "final_divergence": log.final_divergence,
                "convergence_round": log.convergence_round,
                "max_rounds": log.max_rounds,
                "convergence_threshold": log.convergence_threshold,
                "total_rounds": len(log.rounds),
                "created_at": log.created_at,
                "persisted_at": log.persisted_at,
                "immutable_hash": log.immutable_hash,
            }

            # 裁决信息 (Summary 级别也包含)
            if log.adjudicator_verdict:
                result["adjudicator_verdict"] = log.adjudicator_verdict.model_dump()

            # 结果影响 (Summary 级别也包含)
            result["outcome"] = log.outcome.model_dump()

            if level == LogVerbosity.SUMMARY:
                return result

            # Full 级别: + 轮次 + 资源 + Pre-Debate
            result["pre_debate"] = log.pre_debate.model_dump()
            result["rounds"] = [r.model_dump() for r in log.rounds]
            result["divergence_curve"] = list(log.divergence_curve)
            result["resource_usage"] = log.resource_usage.model_dump()

            if level == LogVerbosity.FULL:
                return result

            # Debug 级别: + 脱敏 Prompt
            result["debug_prompts"] = list(log.debug_prompts)

            return result

    # ==========================================================
    # 查询
    # ==========================================================

    def get_log(self, debate_log_id: str) -> DebateLog:
        """获取辩论日志.

        Raises:
            DebateLogNotFoundError: 日志不存在
        """
        with self._lock:
            log = self._logs.get(debate_log_id)
            if log is None:
                raise DebateLogNotFoundError(debate_log_id)
            return log

    def get_by_debate(self, debate_id: str) -> DebateLog | None:
        """按辩论 ID 查询日志."""
        with self._lock:
            log_id = self._debate_index.get(debate_id)
            if log_id:
                return self._logs.get(log_id)
            return None

    def get_by_task(self, task_id: str) -> list[DebateLog]:
        """按任务 ID 查询日志列表."""
        with self._lock:
            ids = self._task_index.get(task_id, [])
            return [self._logs[lid] for lid in ids if lid in self._logs]

    def list_logs(
        self,
        converged: bool | None = None,
        verbosity: LogVerbosity | None = None,
        limit: int = 100,
    ) -> list[DebateLog]:
        """列出辩论日志.

        Args:
            converged: 按收敛状态筛选 (None=全部)
            verbosity: 按日志级别筛选 (None=全部)
            limit: 最多返回数

        Returns:
            日志列表
        """
        with self._lock:
            results = []
            for log in self._logs.values():
                if converged is not None and log.convergence_reached != converged:
                    continue
                if verbosity is not None and log.verbosity != verbosity:
                    continue
                results.append(log)
                if len(results) >= limit:
                    break
            return results

    # ==========================================================
    # 统计
    # ==========================================================

    def statistics(self) -> dict[str, Any]:
        """获取辩论日志统计信息."""
        with self._lock:
            total = len(self._logs)
            if total == 0:
                return {"total": 0}

            converged_count = sum(1 for l in self._logs.values() if l.convergence_reached)
            force_resolved = sum(
                1 for l in self._logs.values()
                if l.convergence_status == ConvergenceStatus.FORCE_RESOLVED
            )
            aborted = sum(
                1 for l in self._logs.values()
                if l.convergence_status == ConvergenceStatus.ABORTED
            )

            by_verbosity: dict[str, int] = {}
            by_status: dict[str, int] = {}
            total_rounds = 0
            total_tokens = 0
            total_cost = 0.0
            divergence_values: list[float] = []

            for log in self._logs.values():
                v = log.verbosity.value
                by_verbosity[v] = by_verbosity.get(v, 0) + 1
                s = log.convergence_status.value
                by_status[s] = by_status.get(s, 0) + 1
                total_rounds += len(log.rounds)
                total_tokens += log.resource_usage.total_tokens
                total_cost += log.resource_usage.estimated_cost
                if log.final_divergence > 0:
                    divergence_values.append(log.final_divergence)

            avg_divergence = (
                sum(divergence_values) / len(divergence_values)
                if divergence_values
                else 0.0
            )
            avg_rounds = total_rounds / total if total > 0 else 0.0
            convergence_rate = converged_count / total if total > 0 else 0.0

            return {
                "total": total,
                "convergence_rate": round(convergence_rate, 4),
                "converged": converged_count,
                "force_resolved": force_resolved,
                "aborted": aborted,
                "by_verbosity": by_verbosity,
                "by_status": by_status,
                "avg_rounds": round(avg_rounds, 2),
                "avg_final_divergence": round(avg_divergence, 4),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
            }

    # ==========================================================
    # 清空 (测试用)
    # ==========================================================

    def clear(self) -> None:
        """清空所有日志."""
        with self._lock:
            self._logs.clear()
            self._debate_index.clear()
            self._task_index.clear()


__all__ = [
    "DebateLogger",
    "DivergenceCalculator",
    "PromptSanitizer",
]
