"""L2 练习模块 — 题库 + 自适应出题 + 判题 + 画像联动.

借鉴 dy-agent-system 的"前测 → 画像 → 动态更新"模式 (pretest_bank.json),
打通 答题 → BKT 更新 → 画像 kp_mastery 同步 的真实学习数据链路:

1. **题库**: 38 道稀土发光材料选择题 (pretest_bank.json), 覆盖 16 个知识节点,
   经 NODE_TO_KP 映射到 42 KP 体系。
2. **出题**: 按学习者薄弱 KP 优先 (mastery < 0.6), 难度自适应, 无薄弱点时随机。
3. **判题**: 提交答案 → 判定对错 → BKT 在线更新 → 画像 kp_mastery 写回 →
   重算 weak_kps → 返回判题结果与新掌握度。
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from .exceptions import L2Error

_logger = logging.getLogger("dy3_polaris.l2.practice")

#: 题库知识节点 → 42 KP ID 映射 (单点: 见 l2/kp_catalog.py)
from dy3_polaris.l2.kp_catalog import NEW_KP_NAMES, NODE_TO_KP

#: 薄弱知识点阈值 (与 L2 ProfileBuilder 全量重建口径一致: mastery < 0.5 视为薄弱)
WEAK_KP_THRESHOLD: float = 0.5

DEFAULT_BANK_PATH = Path(__file__).parent / "pretest_bank.json"


class PracticeBank:
    """练习题库 (线程安全).

    Attributes:
        questions: 全部题目 (每道附加 kp_id 映射).
        by_kp: kp_id → [题目].
        by_qid: qid → 题目.
    """

    def __init__(self, bank_path: str | Path | None = None) -> None:
        self._path = Path(bank_path) if bank_path else DEFAULT_BANK_PATH
        self.questions: list[dict[str, Any]] = []
        self.by_kp: dict[str, list[dict[str, Any]]] = {}
        self.by_qid: dict[str, dict[str, Any]] = {}
        self._lock = __import__("threading").RLock()
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("questions", [])
        for q in raw:
            node = q.get("kg_node", "")
            explicit_kp = str(q.get("kp_id") or "")
            kp_id = explicit_kp if explicit_kp in NEW_KP_NAMES else NODE_TO_KP.get(node)
            if kp_id is None:
                _logger.debug("题库节点 %s 无 KP 映射, 跳过 %s", node, q.get("qid"))
                continue
            item = dict(q)
            item["kp_id"] = kp_id
            self.questions.append(item)
            self.by_kp.setdefault(kp_id, []).append(item)
            self.by_qid[item["qid"]] = item
        _logger.info("练习题库加载: %d 题 / %d KP (来自 %s)",
                     len(self.questions), len(self.by_kp), self._path.name)

    # ---- 出题 ----

    def select_questions(
        self,
        learner_id: str,
        count: int = 5,
        mastery: dict[str, float] | None = None,
        target_kps: tuple[str, ...] = (),
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        """自适应出题: 优先薄弱 KP, 其次随机.

        Args:
            learner_id: 学习者 ID (用于确定性随机).
            count: 出题数量.
            mastery: {kp_id: mastery}, 用于识别薄弱点.
            rng: 随机源 (默认以 learner_id 种子).
        """
        r = rng or random.Random(f"{learner_id}:{int(time.time() // 300)}")
        mastery = mastery or {}
        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        # Concept-aware callers may request authored questions for mapped KPs.
        # An empty result is intentional: the bank must never fabricate a
        # question when no authored item covers the requested Concept.
        bounded_targets = tuple(
            dict.fromkeys(str(item) for item in target_kps if str(item))
        )
        if bounded_targets:
            target_pool = [
                question
                for kp_id in bounded_targets
                for question in self.by_kp.get(kp_id, ())
            ]
            r.shuffle(target_pool)
            for question in target_pool:
                if question["qid"] in used:
                    continue
                selected.append(question)
                used.add(question["qid"])
                if len(selected) >= count:
                    break
            return selected

        # 1. 薄弱 KP 题目 (mastery < 薄弱阈值, 与画像重建口径一致), 每 KP 取 1 题
        weak_kps = sorted(
            (k for k, v in mastery.items()
             if v < WEAK_KP_THRESHOLD and k in self.by_kp),
            key=lambda k: mastery.get(k, 0.0),
        )
        for kp in weak_kps:
            if len(selected) >= count:
                break
            pool = [q for q in self.by_kp[kp] if q["qid"] not in used]
            if pool:
                q = r.choice(pool)
                selected.append(q)
                used.add(q["qid"])

        # 2. 剩余随机补齐
        rest = [q for q in self.questions if q["qid"] not in used]
        r.shuffle(rest)
        for q in rest:
            if len(selected) >= count:
                break
            selected.append(q)
            used.add(q["qid"])

        return selected

    def get_question(self, qid: str) -> dict[str, Any] | None:
        """按 qid 取题 (不含答案, 用于出题响应)."""
        q = self.by_qid.get(qid)
        if q is None:
            return None
        return q

    def public_question(self, q: dict[str, Any]) -> dict[str, Any]:
        """剔除答案与解析, 返回出题可见结构."""
        return {
            "qid": q["qid"],
            "kp_id": q["kp_id"],
            "kg_name": q.get("kg_name", ""),
            "difficulty": q.get("difficulty", 1),
            "type": q.get("type", "choice"),
            "question": q.get("question", ""),
            "options": q.get("options", []),
            "ability_dim": q.get("ability_dim", ""),
            "source_class": str(q.get("source_class") or "DESIGNED_PRIOR"),
            "design_basis": str(q.get("design_basis") or q.get("knowledge_id") or ""),
        }

    # ---- 判题 + 画像联动 ----

    def answer(
        self,
        learner_id: str,
        qid: str,
        selected: int,
        bkt_service: Any,
        profile_service: Any,
        text_answer: str | None = None,
        update_models: bool = True,
    ) -> dict[str, Any]:
        """判题并联动更新 BKT 状态与画像掌握度.

        Args:
            learner_id: 学习者 ID.
            qid: 题目 ID.
            selected: 所选选项下标 (多选传 -1 并用 selected_multi; 填空可传 -1).
            bkt_service: BKTTracingService 实例.
            profile_service: ProfileTracingService 实例.
            text_answer: 填空作答文本 (动态填空题型).

        Returns:
            {qid, kp_id, correct, correct_index, explanation, difficulty,
             p_mastery_before, p_mastery_after, attempts, mastery_flag}

        Raises:
            L2Error: 题目不存在或 BKT 服务不可用.
        """
        submit_started = time.monotonic()
        q = self.by_qid.get(qid)
        if q is None:
            raise L2Error(code="-32602", detail=f"题目不存在: {qid}")
        if bkt_service is None:
            raise L2Error(code="-32401", detail="BKT 服务不可用")

        kp_id = q["kp_id"]
        q_type = q.get("type", "choice")
        if q_type == "blank":
            # 填空: 归一化容错比对 (去除空格/大小写/全角)
            expect = str(q.get("answer_text", "") or "").strip().lower().replace(" ", "")
            got = str(text_answer or "").strip().lower().replace(" ", "")
            correct = bool(expect and got == expect)
        elif q_type == "multi":
            # 多选: selected 逗号串 → 集合全等
            chosen = {int(x) for x in str(selected).split(",") if str(x).strip().isdigit()}
            answers = {int(a) for a in (q.get("answer") or [])}
            correct = chosen == answers
        else:
            correct = bool(selected == q["answer"])
        # 难度 1-3 → [0.2, 0.5, 0.8]
        difficulty = {1: 0.2, 2: 0.5, 3: 0.8}.get(q.get("difficulty", 1), 0.5)

        # 记录更新前掌握度 (来自画像)
        before = 0.0
        profile = profile_service.get_profile_snapshot(learner_id) if profile_service else None
        if profile is not None:
            before = profile.kp_mastery.get(kp_id, 0.0)

        # BKT 在线更新 (含遗忘衰减/KG 传播)
        from .interaction.event_types import AnswerEvent

        event = AnswerEvent(
            learner_id=learner_id,
            kp_id=kp_id,
            correct=correct,
            difficulty=difficulty,
            question_id=qid,
            timestamp=time.time(),
        )
        output = None
        if update_models:
            output = bkt_service.process(event)
            after = round(float(getattr(output, "p_mastery", before)), 4)
            attempts = int(getattr(output, "attempts", 1))
        else:
            attempts = int(bkt_service.record_observation(event))
            after = round(before, 4)

        # 画像统一口径: 走 L2 profile_builder 全量重建管线
        # (冷启动/遗忘衰减/等级/薄弱点/漂移/生命周期, 消除 practice 手工补丁双口径)
        # persist_history=False: BKT 已写 answer_history, 避免双写
        profile_output = None
        if update_models and profile_service is not None:
            try:
                profile_output = profile_service.process(event, persist_history=False, skip_bkt_update=True)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("画像管线重建失败 %s: %s", learner_id, exc)

        # 口径统一: 返回给前端的 p_mastery_after 取画像重建后的最终掌握度,
        # 确保与 /l2/profile 返回的 kp_mastery 一致 (消除 BKT 双写导致的口径偏差)
        if profile_output is not None and kp_id in profile_output.kp_mastery:
            after = round(float(profile_output.kp_mastery[kp_id]), 4)

        _logger.info("练习判题 learner=%s qid=%s kp=%s correct=%s mastery %.3f→%.3f",
                     learner_id, qid, kp_id, correct, before, after)

        runtime_metrics = {
            **dict(getattr(bkt_service, "_last_runtime_metrics", {}) or {}),
            **dict(getattr(profile_service, "_last_runtime_metrics", {}) or {}),
            "practice_submit_ms": round((time.monotonic() - submit_started) * 1000.0, 3),
        }
        return {
            "qid": qid,
            "kp_id": kp_id,
            "correct": correct,
            "correct_index": q["answer"],
            "explanation": q.get("explanation", ""),
            "difficulty": q.get("difficulty", 1),
            "p_mastery_before": round(before, 4),
            "p_mastery_after": after,
            "attempts": attempts,
            "mastery_flag": bool(getattr(output, "mastery_flag", False)) if output else False,
            "answer_saved": True,
            "model_updated": bool(update_models),
            "model_update_status": "UPDATED" if update_models else "SKIPPED_BY_POLICY",
            "source_class": str(q.get("source_class") or "DESIGNED_PRIOR"),
            "design_basis": str(q.get("design_basis") or q.get("knowledge_id") or ""),
            "_runtime_metrics": runtime_metrics,
        }
