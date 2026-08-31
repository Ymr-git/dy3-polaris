"""L4 决策引擎层 — Faithfulness 评估器与自洽性检查器.

借鉴世界先进方案:
- RAGAS (2024): RAG 系统忠实度评估框架
  - Faithfulness: 答案中的主张是否都能从检索上下文中推断
  - Answer Relevancy: 答案与问题的相关程度
  - Context Precision/Recall: 检索质量评估
- VeReaFine (2025): 迭代验证-推理-精炼 RAG，识别缺失证据
- CISC (2025): 置信度感知自洽性，加权多数投票
- VeriCoT (2025): 神经符号 CoT 验证，逻辑一致性检查

核心职责:
    1. FaithfulnessChecker: 评估生成答案与检索上下文的事实一致性
    2. SelfConsistencyChecker: 评估多路径推理答案的自洽性
    3. ClaimExtractor: 从生成内容中提取原子化主张
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .models import (
    ExecutionResult,
    TaskResult,
    TaskType,
)

logger = logging.getLogger(__name__)


# ============================================================
# 主张提取器
# ============================================================


class ClaimExtractor:
    """原子化主张提取器 (借鉴 FActScore).

    从生成内容中提取可验证的原子化主张，
    包括数值型主张和描述型主张。

    主张类型:
    - numeric: 数值型 (如 "发射波长为 580nm")
    - compositional: 组成型 (如 "YAG:Ce³⁺ 包含铈离子")
    - causal: 因果型 (如 "浓度淬灭导致发光强度下降")
    - comparative: 比较型 (如 "Eu³⁺ 的发射波长比 Tb³⁺ 长")
    - descriptive: 描述型 (其他事实陈述)
    """

    # 数值主张正则: 匹配 "X ...为/是/等于 Y数值 + 单位"
    # 支持离子名(如 Dy3+)和中文术语(如 发射波长)作为主语
    # 使用 .*? 允许主语与动词之间存在修饰语 (如 "Dy3+ 的发射波长为 575nm")
    NUMERIC_PATTERN = re.compile(
        r"([\w\u4e00-\u9fff+]+).*?(?:为|是|等于|约|大约)\s*"
        r"(\d+\.?\d*)\s*(nm|μm|μs|ms|ns|K|℃|%|eV|cm|mol|mg|ppm|wt|at)",
        re.IGNORECASE,
    )

    # 组成主张正则: 匹配 "X 包含/含有/由...组成 Y"
    COMPOSITION_PATTERN = re.compile(
        r"([\w\u4e00-\u9fff+]+?)[\s]*(?:包含|含有|由.*组成|掺杂|激活)[\s]*"
        r"([\w\u4e00-\u9fff+]+)",
        re.IGNORECASE,
    )

    # 因果主张正则: 匹配 "X 导致/引起/使得 Y"
    CAUSAL_PATTERN = re.compile(
        r"([\w\u4e00-\u9fff+]+?)[\s]*(?:导致|引起|使得|造成|引发)[\s]*"
        r"([\w\u4e00-\u9fff+]+)",
        re.IGNORECASE,
    )

    # 比较主张正则: 匹配 "X 比 Y 更/较 + 形容词"
    COMPARATIVE_PATTERN = re.compile(
        r"([\w\u4e00-\u9fff+]+?)[\s]*(?:比)[\s]*([\w\u4e00-\u9fff+]+?)[\s]*"
        r"(?:更|较)[\s]*([\w\u4e00-\u9fff+]+)",
        re.IGNORECASE,
    )

    @classmethod
    def extract(cls, text: str) -> list[dict[str, Any]]:
        """从文本中提取原子化主张.

        Args:
            text: 待提取的文本

        Returns:
            主张列表，每个主张包含 type, subject, value, raw_text
        """
        claims: list[dict[str, Any]] = []

        # 提取数值主张
        for match in cls.NUMERIC_PATTERN.finditer(text):
            claims.append({
                "type": "numeric",
                "subject": match.group(1).strip(),
                "value": match.group(2),
                "unit": match.group(3),
                "raw_text": match.group(0),
            })

        # 提取组成主张
        for match in cls.COMPOSITION_PATTERN.finditer(text):
            claims.append({
                "type": "compositional",
                "subject": match.group(1).strip(),
                "value": match.group(2).strip(),
                "raw_text": match.group(0),
            })

        # 提取因果主张
        for match in cls.CAUSAL_PATTERN.finditer(text):
            claims.append({
                "type": "causal",
                "subject": match.group(1).strip(),
                "value": match.group(2).strip(),
                "raw_text": match.group(0),
            })

        # 提取比较主张
        for match in cls.COMPARATIVE_PATTERN.finditer(text):
            claims.append({
                "type": "comparative",
                "subject": match.group(1).strip(),
                "object": match.group(2).strip(),
                "value": match.group(3).strip(),
                "raw_text": match.group(0),
            })

        # 提取描述型主张 (句子级别，简化版)
        sentences = re.split(r"[。；！？\.\;\!\?]", text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            # 跳过已提取的数值/组成/因果/比较句
            if any(c["raw_text"] in sent for c in claims):
                continue
            # 包含事实性关键词的句子作为描述型主张
            fact_keywords = ["是", "有", "属于", "位于", "发生", "表现", "具有", "呈现"]
            if any(kw in sent for kw in fact_keywords):
                claims.append({
                    "type": "descriptive",
                    "subject": "",
                    "value": sent,
                    "raw_text": sent,
                })

        return claims


# ============================================================
# Faithfulness 评估器
# ============================================================


class FaithfulnessChecker:
    """RAGAS Faithfulness 评估器.

    评估生成答案中的主张是否都能从检索上下文中找到支持。

    核心流程:
    1. 从生成答案中提取原子化主张
    2. 对每个主张，在检索上下文中查找支持证据
    3. 计算 Faithfulness = 被支持的主张数 / 总主张数

    借鉴 RAGAS v0.4 + VeReaFine 迭代验证:
    - 主张级别细粒度评估
    - 缺失证据识别 (用于触发补充检索)
    - 上下文精度与召回评估
    """

    def __init__(
        self,
        *,
        support_threshold: float = 0.6,
        enable_missing_evidence_report: bool = True,
    ) -> None:
        """初始化 Faithfulness 评估器.

        Args:
            support_threshold: 支持度阈值，高于此值认为主张被支持
            enable_missing_evidence_report: 是否生成缺失证据报告
        """
        self._support_threshold = support_threshold
        self._enable_missing_evidence = enable_missing_evidence_report
        self._claim_extractor = ClaimExtractor()

        logger.info(
            "FaithfulnessChecker 初始化 (支持度阈值: %.2f)",
            support_threshold,
        )

    def assess(
        self,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """评估执行结果的 Faithfulness.

        Args:
            execution_result: T3 执行结果

        Returns:
            评估结果字典:
            - faithfulness_score: 忠实度分数 (0~1)
            - answer_relevancy: 答案相关性 (0~1)
            - context_precision: 上下文精度 (0~1)
            - context_recall: 上下文召回 (0~1)
            - total_claims: 总主张数
            - supported_claims: 被支持的主张数
            - unsupported_claims: 未被支持的主张列表
            - missing_evidence: 缺失的证据建议
        """
        # 提取生成答案
        answer_text = self._extract_answer_text(execution_result)
        if not answer_text:
            return self._empty_result("无可评估的答案文本")

        # 提取检索上下文
        context_chunks = self._extract_context_chunks(execution_result)
        if not context_chunks:
            return self._empty_result("无检索上下文可用")

        # 提取原子化主张
        claims = self._claim_extractor.extract(answer_text)
        if not claims:
            return {
                "faithfulness_score": 0.8,
                "answer_relevancy": 0.8,
                "context_precision": 0.8,
                "context_recall": 0.8,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "missing_evidence": [],
                "message": "未提取到可验证主张",
            }

        # 逐条验证主张
        supported = 0
        unsupported_claims: list[dict[str, Any]] = []
        missing_evidence: list[dict[str, Any]] = []

        for claim in claims:
            support_score, supporting_chunks = self._check_claim_support(
                claim, context_chunks
            )

            if support_score >= self._support_threshold:
                supported += 1
            else:
                unsupported_claims.append({
                    "claim": claim,
                    "support_score": round(support_score, 4),
                    "best_chunk": supporting_chunks[0] if supporting_chunks else None,
                })

                if self._enable_missing_evidence:
                    missing_evidence.append({
                        "claim_type": claim["type"],
                        "claim_subject": claim.get("subject", ""),
                        "claim_value": claim.get("value", ""),
                        "suggested_query": self._generate_evidence_query(claim),
                    })

        faithfulness = supported / len(claims) if claims else 0.0

        # 评估答案相关性 (基于答案与查询的语义匹配，简化版)
        relevancy = self._assess_relevancy(execution_result, answer_text)

        # 评估上下文精度 (检索到的上下文有多少被实际使用)
        context_precision = self._assess_context_precision(
            context_chunks, claims
        )

        # 评估上下文召回 (需要回答的信息是否都被检索到)
        context_recall = self._assess_context_recall(
            claims, supported
        )

        return {
            "faithfulness_score": round(faithfulness, 4),
            "answer_relevancy": round(relevancy, 4),
            "context_precision": round(context_precision, 4),
            "context_recall": round(context_recall, 4),
            "total_claims": len(claims),
            "supported_claims": supported,
            "unsupported_claims": unsupported_claims,
            "missing_evidence": missing_evidence,
        }

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    @staticmethod
    def _extract_answer_text(result: ExecutionResult) -> str:
        """从执行结果中提取生成答案文本."""
        parts: list[str] = []

        for tr in result.get_results_by_type(TaskType.SYNTHESIZE):
            summary = tr.output.get("summary", "")
            if summary:
                parts.append(summary)

        for tr in result.get_results_by_type(TaskType.REASON):
            answers = tr.output.get("answers", [])
            for ans in answers:
                if isinstance(ans, dict):
                    text = ans.get("text") or ans.get("value") or str(ans)
                    parts.append(text)
                elif isinstance(ans, str):
                    parts.append(ans)

        return "\n".join(parts)

    @staticmethod
    def _extract_context_chunks(result: ExecutionResult) -> list[dict[str, Any]]:
        """从执行结果中提取检索上下文块."""
        chunks: list[dict[str, Any]] = []

        for tr in result.get_results_by_type(TaskType.RETRIEVE):
            results = tr.output.get("results", [])
            for r in results:
                if isinstance(r, dict):
                    chunks.append({
                        "chunk_id": r.get("chunk_id", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                        "source": r.get("source_type", tr.task_id),
                    })

        return chunks

    @staticmethod
    def _check_claim_support(
        claim: dict[str, Any],
        context_chunks: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """检查单个主张是否被上下文支持.

        Returns:
            (支持度分数, 支持该主张的上下文块列表)
        """
        claim_text = claim.get("raw_text", "")
        claim_subject = claim.get("subject", "")
        claim_value = claim.get("value", "")

        if not claim_text:
            return 0.0, []

        supporting: list[dict[str, Any]] = []
        best_score = 0.0

        for chunk in context_chunks:
            content = chunk.get("content", "")
            if not content:
                continue

            # 简化版支持度计算: 基于关键词匹配
            score = 0.0

            # 主张主体在上下文中出现
            if claim_subject and claim_subject in content:
                score += 0.3

            # 主张值在上下文中出现
            if claim_value and str(claim_value) in content:
                score += 0.3

            # 主张原文的关键部分在上下文中出现
            claim_keywords = [
                w for w in claim_text.split()
                if len(w) > 1
            ]
            if claim_keywords:
                matched = sum(1 for kw in claim_keywords if kw in content)
                score += 0.4 * (matched / len(claim_keywords))

            # 数值型主张特殊处理: 检查数值是否在上下文中出现
            if claim["type"] == "numeric":
                value_str = str(claim.get("value", ""))
                if value_str in content:
                    score = max(score, 0.7)

            if score >= 0.3:
                supporting.append(chunk)
                best_score = max(best_score, score)

        return best_score, supporting

    @staticmethod
    def _generate_evidence_query(claim: dict[str, Any]) -> str:
        """根据未支持的主张生成补充检索查询."""
        claim_type = claim.get("type", "")
        subject = claim.get("subject", "")
        value = claim.get("value", "")

        if claim_type == "numeric":
            return f"{subject} {value} {claim.get('unit', '')} 标准值 文献"
        if claim_type == "compositional":
            return f"{subject} {value} 组成 结构"
        if claim_type == "causal":
            return f"{subject} {value} 机理 原因"
        if claim_type == "comparative":
            obj = claim.get("object", "")
            return f"{subject} {obj} {value} 对比"
        return f"{subject} {value}"

    @staticmethod
    def _assess_relevancy(
        result: ExecutionResult,
        answer_text: str,
    ) -> float:
        """评估答案相关性 (简化版)."""
        # 基于推理结果的置信度和答案长度
        reason_results = result.get_results_by_type(TaskType.REASON)
        if not reason_results:
            return 0.5

        avg_conf = sum(r.confidence for r in reason_results) / len(reason_results)

        # 答案长度因素: 过短可能信息不足，过长可能跑题
        length_factor = 1.0
        if len(answer_text) < 20:
            length_factor = 0.7
        elif len(answer_text) > 2000:
            length_factor = 0.9

        return avg_conf * length_factor

    @staticmethod
    def _assess_context_precision(
        context_chunks: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> float:
        """评估上下文精度 — 检索到的上下文有多少被实际使用."""
        if not context_chunks:
            return 0.0

        used_count = 0
        for chunk in context_chunks:
            content = chunk.get("content", "")
            # 检查该上下文块是否包含任何主张的关键信息
            for claim in claims:
                subject = claim.get("subject", "")
                value = str(claim.get("value", ""))
                if (subject and subject in content) or (value and value in content):
                    used_count += 1
                    break

        return used_count / len(context_chunks)

    @staticmethod
    def _assess_context_recall(
        claims: list[dict[str, Any]],
        supported_count: int,
    ) -> float:
        """评估上下文召回 — 需要回答的信息是否都被检索到."""
        if not claims:
            return 1.0
        return supported_count / len(claims)

    @staticmethod
    def _empty_result(message: str) -> dict[str, Any]:
        """生成空结果."""
        return {
            "faithfulness_score": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "missing_evidence": [],
            "message": message,
        }


# ============================================================
# 自洽性检查器
# ============================================================


class SelfConsistencyChecker:
    """自洽性检查器 (借鉴 CISC + VeriCoT).

    评估多路径推理答案的自洽性:
    1. 答案一致性: 多条推理路径是否得出相同答案
    2. 推理链逻辑一致性: 推理步骤是否互相矛盾
    3. 置信度校准: 高置信度答案是否确实更可靠

    借鉴 CISC 的加权多数投票:
    - 每条推理路径按置信度加权
    - 高置信度路径获得更大投票权
    """

    def __init__(
        self,
        *,
        consistency_threshold: float = 0.7,
        enable_logic_check: bool = True,
    ) -> None:
        """初始化自洽性检查器.

        Args:
            consistency_threshold: 自洽性阈值
            enable_logic_check: 是否启用逻辑一致性检查
        """
        self._consistency_threshold = consistency_threshold
        self._enable_logic_check = enable_logic_check

        logger.info(
            "SelfConsistencyChecker 初始化 (阈值: %.2f)",
            consistency_threshold,
        )

    def assess(
        self,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """评估执行结果的自洽性.

        Args:
            execution_result: T3 执行结果

        Returns:
            评估结果字典:
            - consistency_score: 自洽性分数 (0~1)
            - answer_agreement: 答案一致性 (0~1)
            - logic_consistency: 逻辑一致性 (0~1)
            - confidence_calibration: 置信度校准 (0~1)
            - contradictions: 发现的矛盾列表
            - dominant_answer: 主导答案
        """
        reason_results = execution_result.get_results_by_type(TaskType.REASON)

        if not reason_results:
            return {
                "consistency_score": 0.8,
                "answer_agreement": 0.8,
                "logic_consistency": 0.8,
                "confidence_calibration": 0.8,
                "contradictions": [],
                "dominant_answer": None,
                "message": "无推理结果，使用默认值",
            }

        if len(reason_results) == 1:
            # 单路径推理，仅检查内部逻辑一致性
            tr = reason_results[0]
            logic_score = self._check_internal_logic(tr)
            return {
                "consistency_score": tr.confidence * 0.5 + logic_score * 0.5,
                "answer_agreement": 1.0,  # 单路径，完全一致
                "logic_consistency": logic_score,
                "confidence_calibration": tr.confidence,
                "contradictions": [],
                "dominant_answer": self._extract_dominant_answer(tr),
                "message": "单路径推理",
            }

        # 多路径推理: 检查答案一致性
        answer_agreement, dominant_answer, contradictions = (
            self._check_answer_agreement(reason_results)
        )

        # 逻辑一致性
        logic_score = 1.0
        if self._enable_logic_check:
            logic_scores = [
                self._check_internal_logic(tr) for tr in reason_results
            ]
            logic_score = sum(logic_scores) / len(logic_scores) if logic_scores else 1.0

        # 置信度校准: 高置信度路径的答案是否与主导答案一致
        calibration = self._check_confidence_calibration(
            reason_results, dominant_answer
        )

        # 综合自洽性
        consistency = (
            0.4 * answer_agreement
            + 0.3 * logic_score
            + 0.3 * calibration
        )

        return {
            "consistency_score": round(consistency, 4),
            "answer_agreement": round(answer_agreement, 4),
            "logic_consistency": round(logic_score, 4),
            "confidence_calibration": round(calibration, 4),
            "contradictions": contradictions,
            "dominant_answer": dominant_answer,
            "num_paths": len(reason_results),
        }

    # --------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------

    def _check_answer_agreement(
        self,
        reason_results: list[TaskResult],
    ) -> tuple[float, dict[str, Any] | None, list[dict[str, Any]]]:
        """检查多路径推理的答案一致性 (借鉴 CISC 加权投票).

        Returns:
            (一致性分数, 主导答案, 矛盾列表)
        """
        # 收集所有路径的答案
        answer_votes: dict[str, float] = {}  # answer_text -> weighted_vote
        answer_details: dict[str, dict[str, Any]] = {}

        for tr in reason_results:
            answers = tr.output.get("answers", [])
            for ans in answers:
                if isinstance(ans, dict):
                    text = ans.get("text") or ans.get("value") or str(ans)
                else:
                    text = str(ans)

                text = str(text).strip()
                if not text:
                    continue

                # CISC 加权投票: 高置信度路径权重更大
                weight = tr.confidence
                answer_votes[text] = answer_votes.get(text, 0.0) + weight

                if text not in answer_details:
                    answer_details[text] = {
                        "text": text,
                        "total_weight": 0.0,
                        "sources": [],
                    }
                answer_details[text]["total_weight"] += weight
                answer_details[text]["sources"].append({
                    "task_id": tr.task_id,
                    "confidence": tr.confidence,
                })

        if not answer_votes:
            return 0.5, None, []

        # 找出主导答案
        dominant_text = max(answer_votes, key=answer_votes.get)
        dominant_weight = answer_votes[dominant_text]
        total_weight = sum(answer_votes.values())

        # 一致性分数 = 主导答案的权重占比
        agreement = dominant_weight / total_weight if total_weight > 0 else 0.0

        # 识别矛盾
        contradictions: list[dict[str, Any]] = []
        if len(answer_votes) > 1:
            for text, weight in answer_votes.items():
                if text != dominant_text and weight / total_weight > 0.2:
                    contradictions.append({
                        "dominant_answer": dominant_text,
                        "conflicting_answer": text,
                        "dominant_weight": round(dominant_weight, 4),
                        "conflicting_weight": round(weight, 4),
                        "conflict_ratio": round(weight / dominant_weight, 4),
                    })

        dominant_answer = answer_details.get(dominant_text)
        return agreement, dominant_answer, contradictions

    @staticmethod
    def _check_internal_logic(task_result: TaskResult) -> float:
        """检查单条推理路径的内部逻辑一致性.

        简化版 VeriCoT: 检查推理链中是否存在明显矛盾。
        """
        chain = task_result.output.get("reasoning_chain", [])
        if not chain:
            return 0.7

        # 检查推理链长度 (过短可能逻辑不充分)
        length_score = min(1.0, len(chain) / 3.0)

        # 检查推理链中是否有否定词互相矛盾 (简化版)
        negation_patterns = ["不", "非", "无", "没有", "否", "错"]
        affirmation_patterns = ["是", "有", "正确", "成立"]

        has_negation = False
        has_affirmation = False
        for step in chain:
            step_text = str(step)
            if any(neg in step_text for neg in negation_patterns):
                has_negation = True
            if any(aff in step_text for aff in affirmation_patterns):
                has_affirmation = True

        # 同时出现肯定和否定不一定是矛盾，但降低分数
        conflict_penalty = 0.1 if (has_negation and has_affirmation) else 0.0

        # 基于置信度
        conf_score = task_result.confidence

        return max(0.0, min(1.0, 0.4 * length_score + 0.6 * conf_score - conflict_penalty))

    @staticmethod
    def _check_confidence_calibration(
        reason_results: list[TaskResult],
        dominant_answer: dict[str, Any] | None,
    ) -> float:
        """检查置信度校准 — 高置信度路径是否与主导答案一致."""
        if not dominant_answer:
            return 0.5

        dominant_text = dominant_answer.get("text", "")

        # 按置信度排序
        sorted_results = sorted(
            reason_results,
            key=lambda r: r.confidence,
            reverse=True,
        )

        # 检查高置信度路径是否支持主导答案
        top_k = min(3, len(sorted_results))
        top_results = sorted_results[:top_k]

        agreement_count = 0
        for tr in top_results:
            answers = tr.output.get("answers", [])
            for ans in answers:
                text = ans.get("text") or ans.get("value") or str(ans) if isinstance(ans, dict) else str(ans)
                if str(text).strip() == dominant_text:
                    agreement_count += 1
                    break

        return agreement_count / top_k if top_k > 0 else 0.5

    @staticmethod
    def _extract_dominant_answer(task_result: TaskResult) -> dict[str, Any] | None:
        """从单条推理结果中提取主导答案."""
        answers = task_result.output.get("answers", [])
        if not answers:
            return None

        ans = answers[0]
        if isinstance(ans, dict):
            return {
                "text": ans.get("text") or ans.get("value") or str(ans),
                "total_weight": task_result.confidence,
                "sources": [{"task_id": task_result.task_id, "confidence": task_result.confidence}],
            }
        return {
            "text": str(ans),
            "total_weight": task_result.confidence,
            "sources": [{"task_id": task_result.task_id, "confidence": task_result.confidence}],
        }


__all__ = [
    "ClaimExtractor",
    "FaithfulnessChecker",
    "SelfConsistencyChecker",
]
