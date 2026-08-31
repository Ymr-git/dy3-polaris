"""CC1 四层反幻觉评审引擎 — 评审状态机与自纠回路.

实现设计文档中定义的状态转换流程:

    IDLE → L1_FACT → L2_LOGIC → L3_NUMERICAL → L4_PROVENANCE → COMPOSITE_SCORE
                                                ↓
                    ┌──────────── SELF_CORRECT ← (Block/Flag)
                    │                        ↓
                    │          AgentOutput (第 1/2 次自纠)
                    │                        ↓
                    └──── 重新评审 ←──────────┘
                                   ↓ (第 3 次失败)
                              ESCALATE

状态转换规则:
- L1 Block → SELF_CORRECT (触发自纠回路)
- L2 Block → SELF_CORRECT
- L3 Block → SELF_CORRECT
- L4 Flag  → SELF_CORRECT
- COMPOSITE_SCORE: score >= 85 → PASS, 60-85 → FLAG, < 60 → BLOCK
- SELF_CORRECT: 第 1/2 次自纠 → 重新评审; 第 3 次 → ESCALATE/BLOCK

融合世界先进方案:
- CoVe (Chain-of-Verification): 自纠回路中验证-修正-再验证
- SelfCheckGPT: 多次采样自洽性检查
- Guardrails AI: 可配置的纠正策略
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .layers import ReviewLayerType


# ============================================================
# 枚举定义
# ============================================================


class ReviewState(str, Enum):
    """评审状态机状态."""

    IDLE = "idle"
    L1_FACT = "l1_fact"
    L2_LOGIC = "l2_logic"
    L3_NUMERICAL = "l3_numerical"
    L4_PROVENANCE = "l4_provenance"
    SELF_CORRECT = "self_correct"
    COMPOSITE_SCORE = "composite_score"
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"
    ESCALATE = "escalate"


class ReviewVerdict(str, Enum):
    """评审判决结果 — 三级."""

    PASS = "pass"      # 通过: 无问题或问题轻微
    FLAG = "flag"      # 警告: 有问题但可修正
    BLOCK = "block"    # 阻断: 严重问题, 不可通过


# ============================================================
# 评审状态机
# ============================================================


class ReviewStateMachine:
    """评审状态机.

    管理四层递进评审的状态转换, 记录完整状态历史.

    状态转换图::

        IDLE → L1_FACT → L2_LOGIC → L3_NUMERICAL → L4_PROVENANCE → COMPOSITE_SCORE
                                                        ↓
                    Block/Flag → SELF_CORRECT → (重试 or ESCALATE)
    """

    #: 合法的前驱-后继转换关系
    _TRANSITIONS: dict[ReviewState, set[ReviewState]] = {
        ReviewState.IDLE: {
            ReviewState.L1_FACT,
            ReviewState.COMPOSITE_SCORE,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
        },
        ReviewState.L1_FACT: {
            ReviewState.L2_LOGIC,
            ReviewState.SELF_CORRECT,
            ReviewState.L3_NUMERICAL,
            ReviewState.L4_PROVENANCE,
            ReviewState.COMPOSITE_SCORE,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
        },
        ReviewState.L2_LOGIC: {
            ReviewState.L3_NUMERICAL,
            ReviewState.SELF_CORRECT,
            ReviewState.L4_PROVENANCE,
            ReviewState.COMPOSITE_SCORE,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
        },
        ReviewState.L3_NUMERICAL: {
            ReviewState.L4_PROVENANCE,
            ReviewState.SELF_CORRECT,
            ReviewState.COMPOSITE_SCORE,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
        },
        ReviewState.L4_PROVENANCE: {
            ReviewState.COMPOSITE_SCORE,
            ReviewState.SELF_CORRECT,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
        },
        ReviewState.SELF_CORRECT: {
            ReviewState.L1_FACT,
            ReviewState.L2_LOGIC,
            ReviewState.L3_NUMERICAL,
            ReviewState.L4_PROVENANCE,
            ReviewState.COMPOSITE_SCORE,
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
            ReviewState.ESCALATE,
        },
        ReviewState.COMPOSITE_SCORE: {
            ReviewState.PASS,
            ReviewState.FLAG,
            ReviewState.BLOCK,
            ReviewState.SELF_CORRECT,
        },
        ReviewState.PASS: set(),
        ReviewState.FLAG: {
            ReviewState.SELF_CORRECT,
            ReviewState.PASS,
            ReviewState.BLOCK,
        },
        ReviewState.BLOCK: {
            ReviewState.SELF_CORRECT,
            ReviewState.ESCALATE,
            ReviewState.PASS,
        },
        ReviewState.ESCALATE: set(),
    }

    def __init__(self) -> None:
        self._current_state: ReviewState = ReviewState.IDLE
        self._history: list[dict[str, Any]] = []
        self._warnings: set[ReviewLayerType] = set()
        self._blocks: set[ReviewLayerType] = set()

    @property
    def current_state(self) -> ReviewState:
        """当前状态."""
        return self._current_state

    @property
    def history(self) -> list[dict[str, Any]]:
        """状态转换历史."""
        return self._history

    @property
    def is_terminal(self) -> bool:
        """是否处于终态."""
        return self._current_state in (
            ReviewState.PASS,
            ReviewState.BLOCK,
            ReviewState.ESCALATE,
        )

    def transition(
        self,
        new_state: ReviewState,
        *,
        verdict: ReviewVerdict | None = None,
        score: float | None = None,
        **metadata: Any,
    ) -> None:
        """状态转换.

        Args:
            new_state: 目标状态
            verdict: 评审判决 (可选)
            score: 评分 (可选)
            **metadata: 附加元数据
        """
        from_state = self._current_state
        self._current_state = new_state

        # 记录警告
        if verdict == ReviewVerdict.FLAG and from_state in (
            ReviewState.L1_FACT,
            ReviewState.L2_LOGIC,
            ReviewState.L3_NUMERICAL,
            ReviewState.L4_PROVENANCE,
        ):
            layer_map = {
                ReviewState.L1_FACT: ReviewLayerType.L1_FACT,
                ReviewState.L2_LOGIC: ReviewLayerType.L2_LOGIC,
                ReviewState.L3_NUMERICAL: ReviewLayerType.L3_NUMERICAL,
                ReviewState.L4_PROVENANCE: ReviewLayerType.L4_PROVENANCE,
            }
            if from_state in layer_map:
                self._warnings.add(layer_map[from_state])

        # 记录阻断
        if verdict == ReviewVerdict.BLOCK and from_state in (
            ReviewState.L1_FACT,
            ReviewState.L2_LOGIC,
            ReviewState.L3_NUMERICAL,
            ReviewState.L4_PROVENANCE,
        ):
            layer_map = {
                ReviewState.L1_FACT: ReviewLayerType.L1_FACT,
                ReviewState.L2_LOGIC: ReviewLayerType.L2_LOGIC,
                ReviewState.L3_NUMERICAL: ReviewLayerType.L3_NUMERICAL,
                ReviewState.L4_PROVENANCE: ReviewLayerType.L4_PROVENANCE,
            }
            if from_state in layer_map:
                self._blocks.add(layer_map[from_state])

        # 记录历史
        entry: dict[str, Any] = {
            "from": from_state,
            "to": new_state,
            "verdict": verdict,
            "score": score,
            "timestamp": time.time(),
        }
        entry.update(metadata)
        self._history.append(entry)

    def has_warning(self, layer_type: ReviewLayerType) -> bool:
        """检查指定层是否有警告."""
        return layer_type in self._warnings

    def has_block(self, layer_type: ReviewLayerType) -> bool:
        """检查指定层是否有阻断."""
        return layer_type in self._blocks

    def reset(self) -> None:
        """重置状态机."""
        self._current_state = ReviewState.IDLE
        self._history.clear()
        self._warnings.clear()
        self._blocks.clear()


# ============================================================
# 自纠回路
# ============================================================


@dataclass
class SelfCorrectionRecord:
    """自纠记录.

    Attributes:
        attempt_number: 尝试次数 (1-based)
        issues: 发现的问题列表
        suggestions: 修正建议列表
        timestamp: 记录时间
    """

    attempt_number: int
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    corrected_output: str | None = None


class SelfCorrectionLoop:
    """自纠回路.

    管理自纠重试逻辑: 最多 max_attempts 次自纠,
    超过后升级到 ESCALATE 状态.

    融合 CoVe (Chain-of-Verification) 策略:
    1. 识别问题 (issues)
    2. 生成修正建议 (suggestions)
    3. 应用修正并重新验证
    4. 若修正成功则结束, 否则继续重试
    """

    def __init__(self, max_attempts: int = 2) -> None:
        self._max_attempts = max_attempts
        self._attempts: int = 0
        self._records: list[SelfCorrectionRecord] = []
        self._results: dict[int, dict[str, Any]] = {}
        self._resolved: bool = False

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def attempts(self) -> int:
        """已尝试次数."""
        return self._attempts

    @property
    def can_retry(self) -> bool:
        """是否可以继续重试."""
        return self._attempts < self._max_attempts and not self._resolved

    @property
    def needs_escalation(self) -> bool:
        """是否需要升级 (已用完所有重试且未解决)."""
        return self._attempts >= self._max_attempts and not self._resolved

    @property
    def is_resolved(self) -> bool:
        """是否已解决."""
        return self._resolved

    @property
    def history(self) -> list[dict[str, Any]]:
        """自纠历史记录."""
        result: list[dict[str, Any]] = []
        for record in self._records:
            entry: dict[str, Any] = {
                "attempt": record.attempt_number,
                "issues": record.issues,
                "suggestions": record.suggestions,
                "success": self._results.get(record.attempt_number, {}).get("success", record.success),
                "corrected_output": self._results.get(record.attempt_number, {}).get("corrected_output", record.corrected_output),
            }
            result.append(entry)
        return result

    def start_correction(
        self,
        issues: list[str],
        suggestions: list[str],
    ) -> SelfCorrectionRecord:
        """开始一次自纠.

        Args:
            issues: 发现的问题列表
            suggestions: 修正建议列表

        Returns:
            自纠记录
        """
        self._attempts += 1
        record = SelfCorrectionRecord(
            attempt_number=self._attempts,
            issues=list(issues),
            suggestions=list(suggestions),
        )
        self._records.append(record)
        return record

    def record_result(
        self,
        attempt_number: int,
        *,
        success: bool,
        corrected_output: str | None = None,
    ) -> None:
        """记录自纠结果.

        Args:
            attempt_number: 尝试编号
            success: 是否成功
            corrected_output: 修正后的输出
        """
        self._results[attempt_number] = {
            "success": success,
            "corrected_output": corrected_output,
        }
        # 更新对应记录
        for record in self._records:
            if record.attempt_number == attempt_number:
                record.success = success
                record.corrected_output = corrected_output
                break
        if success:
            self._resolved = True

    def get_latest_issues(self) -> list[str]:
        """获取最新一次自纠的问题."""
        if not self._records:
            return []
        return self._records[-1].issues

    def get_latest_suggestions(self) -> list[str]:
        """获取最新一次自纠的建议."""
        if not self._records:
            return []
        return self._records[-1].suggestions
