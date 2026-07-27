"""CC1 防幻觉层 — 验证存储.

存储验证报告和幻觉检测记录，提供查询、统计和导出能力。
遵循 L0 PolicyStore / L6 ProvenanceStore 的设计模式：
线程安全（RLock）、容量控制（FIFO 淘汰）、多维查询。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .models import (
    HallucinationRecord,
    HallucinationSeverity,
    VerificationReport,
    VerificationStatus,
    VerdictAction,
)


class VerificationStore:
    """验证存储.

    线程安全的验证报告和幻觉记录存储。

    Attributes:
        max_reports: 最大报告数（FIFO 淘汰）
        max_records: 最大幻觉记录数
    """

    _DEFAULT_MAX_REPORTS = 1000
    _DEFAULT_MAX_RECORDS = 500

    def __init__(
        self,
        max_reports: int = _DEFAULT_MAX_REPORTS,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        self._max_reports = max_reports
        self._max_records = max_records
        self._reports: dict[str, VerificationReport] = {}
        self._records: dict[str, HallucinationRecord] = {}
        self._report_order: list[str] = []
        self._record_order: list[str] = []
        self._lock = threading.RLock()

        # 统计计数器
        self._total_verifications = 0
        self._passed_count = 0
        self._failed_count = 0
        self._degraded_count = 0
        self._refused_count = 0
        self._hallucination_count = 0

    # --------------------------------------------------------
    # 报告管理
    # --------------------------------------------------------

    def add_report(self, report: VerificationReport) -> None:
        """添加验证报告."""
        with self._lock:
            # 容量控制
            if len(self._reports) >= self._max_reports:
                oldest_id = self._report_order.pop(0)
                self._reports.pop(oldest_id, None)

            self._reports[report.report_id] = report
            self._report_order.append(report.report_id)
            self._total_verifications += 1

            # 更新统计
            if report.status == VerificationStatus.PASSED:
                self._passed_count += 1
            elif report.status == VerificationStatus.FAILED:
                self._failed_count += 1
            elif report.status == VerificationStatus.DEGRADED:
                self._degraded_count += 1
            elif report.status == VerificationStatus.REFUSED:
                self._refused_count += 1

    def get_report(self, report_id: str) -> VerificationReport | None:
        """获取验证报告."""
        with self._lock:
            return self._reports.get(report_id)

    def query_reports(
        self,
        *,
        agent_id: str | None = None,
        status: VerificationStatus | None = None,
        action: VerdictAction | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
    ) -> list[VerificationReport]:
        """多条件查询验证报告."""
        with self._lock:
            results = list(self._reports.values())

        if agent_id is not None:
            results = [r for r in results if r.agent_id == agent_id]
        if status is not None:
            results = [r for r in results if r.status == status]
        if action is not None:
            results = [r for r in results if r.action == action]
        if min_score is not None:
            results = [r for r in results if r.overall_score >= min_score]
        if max_score is not None:
            results = [r for r in results if r.overall_score <= max_score]

        # 按创建时间降序
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    # --------------------------------------------------------
    # 幻觉记录管理
    # --------------------------------------------------------

    def add_record(self, record: HallucinationRecord) -> None:
        """添加幻觉检测记录."""
        with self._lock:
            if len(self._records) >= self._max_records:
                oldest_id = self._record_order.pop(0)
                self._records.pop(oldest_id, None)

            self._records[record.record_id] = record
            self._record_order.append(record.record_id)
            self._hallucination_count += 1

    def get_record(self, record_id: str) -> HallucinationRecord | None:
        """获取幻觉检测记录."""
        with self._lock:
            return self._records.get(record_id)

    def query_records(
        self,
        *,
        agent_id: str | None = None,
        severity: HallucinationSeverity | None = None,
        action: VerdictAction | None = None,
        limit: int = 100,
    ) -> list[HallucinationRecord]:
        """多条件查询幻觉检测记录."""
        with self._lock:
            results = list(self._records.values())

        if agent_id is not None:
            results = [r for r in results if r.agent_id == agent_id]
        if severity is not None:
            results = [r for r in results if r.severity == severity]
        if action is not None:
            results = [r for r in results if r.action_taken == action]

        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    # --------------------------------------------------------
    # 统计与导出
    # --------------------------------------------------------

    @property
    def report_count(self) -> int:
        """报告总数."""
        return len(self._reports)

    @property
    def record_count(self) -> int:
        """幻觉记录总数."""
        return len(self._records)

    @property
    def total_verifications(self) -> int:
        """历史验证总数."""
        return self._total_verifications

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息."""
        with self._lock:
            by_severity: dict[str, int] = {}
            for record in self._records.values():
                key = record.severity.value
                by_severity[key] = by_severity.get(key, 0) + 1

            by_action: dict[str, int] = {}
            for report in self._reports.values():
                key = report.action.value
                by_action[key] = by_action.get(key, 0) + 1

            return {
                "total_verifications": self._total_verifications,
                "current_reports": len(self._reports),
                "passed": self._passed_count,
                "failed": self._failed_count,
                "degraded": self._degraded_count,
                "refused": self._refused_count,
                "hallucination_records": len(self._records),
                "hallucination_total": self._hallucination_count,
                "by_severity": by_severity,
                "by_action": by_action,
            }

    def export_all(self) -> dict[str, Any]:
        """导出全部数据."""
        with self._lock:
            return {
                "reports": [r.model_dump(mode="json") for r in self._reports.values()],
                "records": [r.model_dump(mode="json") for r in self._records.values()],
                "stats": self.get_stats(),
            }

    def export_summary(self) -> dict[str, Any]:
        """导出摘要（仅 ID 和关键指标）."""
        with self._lock:
            return {
                "report_ids": list(self._report_order),
                "record_ids": list(self._record_order),
                "stats": self.get_stats(),
            }

    def clear(self) -> None:
        """清空所有数据."""
        with self._lock:
            self._reports.clear()
            self._records.clear()
            self._report_order.clear()
            self._record_order.clear()
            self._total_verifications = 0
            self._passed_count = 0
            self._failed_count = 0
            self._degraded_count = 0
            self._refused_count = 0
            self._hallucination_count = 0
