"""用户理解服务门面 — 编排提取/蒸馏/推理/提问.

设计:
- profile_store: dict-like {learner_id: UnderstandingProfile} (外部可注入
  InMemoryL2Store 兼容包装, 默认进程内字典).
- extract: 语料 → 信号 → 蒸馏分类 → 推理更新画像.
- ask: 主动提问 (含频率控制, 自动递增计数).
- answer: 用户回答回写 (slot_key 映射到画像字段).
- correct: 用户纠正理解摘要.
- clear: 隐私删除.
- insights: 渐进式揭示摘要 (供前端注入推荐).
- apply_to_snapshot: 将画像写入 LearnerSnapshot.extras["user_profile"].
"""
from __future__ import annotations

import time
from typing import Any

from dy3_polaris.l2.models import LearnerSnapshot

from dy3_polaris.l2.user_understanding.asker import ProactiveAsker
from dy3_polaris.l2.user_understanding.distiller import MemoryDistiller
from dy3_polaris.l2.user_understanding.extractor import CorpusExtractor
from dy3_polaris.l2.user_understanding.inference import ProfileInference
from dy3_polaris.l2.user_understanding.models import (
    UnderstandingProfile,
    UserSignal,
)
from dy3_polaris.l2.user_understanding.privacy import PrivacyGate

_KNOWN_TOPICS = [
    "浓度猝灭", "量子效率", "热猝灭", "发光机理", "能级跃迁", "荧光粉",
    "XRD", "PL光谱", "掺杂", "镝离子", "铕离子", "色度", "显色指数",
    "白光LED", "蓝光", "黄光", "合成方法", "晶体结构",
]

_SLOT_MAP = {
    "direction": {"goal": {"text": "近期学习方向", "type": "short_term"}},
    "vark_pref": {"vark": {"V": 0.5, "A": 0.5}},
    "difficulty_pref": {"preference": {"difficulty": "..."}},
}

_DECLARED_BACKGROUND_SLOTS = {
    "learning_stage",
    "professional_background",
    "domain_experience",
    "learning_goal",
    "representation_preference",
}


class UserUnderstandingService:
    """用户理解服务门面."""

    def __init__(self, profile_store: dict[str, dict[str, Any]] | None = None,
                 known_topics: list[str] | None = None,
                 max_ask_per_session: int = 3,
                 habit_threshold: int = 3,
                 profile_service: Any | None = None) -> None:
        self._store: dict[str, dict[str, Any]] = profile_store if profile_store is not None else {}
        self._profile_service = profile_service
        self._privacy = PrivacyGate()
        self._extractor = CorpusExtractor(known_topics=known_topics or _KNOWN_TOPICS)
        self._distiller = MemoryDistiller(habit_threshold=habit_threshold)
        self._inference = ProfileInference()
        self._asker = ProactiveAsker(max_per_session=max_ask_per_session)

    # ---- 画像读写 ----

    def get_profile(self, learner_id: str) -> UnderstandingProfile | None:
        raw = self._store.get(learner_id)
        if raw is None and self._profile_service is not None:
            try:
                snapshot = self._profile_service.get_profile_snapshot(learner_id)
                extras = dict(getattr(snapshot, "extras", {}) or {}) if snapshot else {}
                persisted = extras.get("user_profile")
                if isinstance(persisted, dict):
                    raw = dict(persisted)
                    self._store[learner_id] = raw
            except Exception:  # noqa: BLE001 - unavailable persistence is non-fatal
                raw = None
        return UnderstandingProfile.from_dict(raw) if raw else None

    def _save(self, prof: UnderstandingProfile) -> None:
        serialized = prof.to_dict()
        self._store[prof.learner_id] = serialized
        service = self._profile_service
        if service is None:
            return
        try:
            snapshot = service.get_profile_snapshot(prof.learner_id)
            if snapshot is None:
                # Metadata-only cold start must remain unknown; creating a
                # profile cannot silently claim beginner mastery or IRT state.
                snapshot = LearnerSnapshot(
                    learner_id=prof.learner_id,
                    snapshot_ts=time.time(),
                    theta=None,
                    level="unknown",
                    learning_style="unknown",
                    bloom_target="unknown",
                    confidence=0.0,
                    extras={"user_profile": serialized},
                )
                store = getattr(service, "store", None)
                save_profile = getattr(store, "save_profile", None)
                if callable(save_profile):
                    save_profile(prof.learner_id, snapshot)
                return
            apply_update = getattr(service, "apply_update", None)
            if callable(apply_update):
                apply_update(
                    prof.learner_id,
                    updates={"extras": {"user_profile": serialized}},
                    expected_version=getattr(snapshot, "version", None),
                )
        except Exception:  # noqa: BLE001 - persistence must not break guidance
            return

    def _ensure(self, learner_id: str) -> UnderstandingProfile:
        p = self.get_profile(learner_id)
        if p is None:
            p = UnderstandingProfile(learner_id=learner_id)
        return p

    # ---- 核心操作 ----

    def extract(self, learner_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """提交对话语料 → 提取并更新画像. 返回提取到的信号列表 (dict)."""
        signals = self._extractor.extract(learner_id, turns)
        if not signals:
            return []
        task_sigs, habit_sigs = self._distiller.distill(signals)
        prof = self._ensure(learner_id)
        # 习惯信号 → habit 记录 → merge (仅 pace/expression/question_style 类)
        for rec in self._distiller.to_habit_records(habit_sigs):
            prof.add_habit(rec)
        # 推理信号 = 任务信号 + habit 中需要直接写入画像的类型 (goal/interest/vark/frustration)
        habit_apply = [
            s for s in habit_sigs
            if s.signal_type.value in ("goal", "interest", "vark", "frustration")
        ]
        self._inference.apply(prof, task_sigs + habit_apply)
        prof.merge_from_habits()
        self._save(prof)
        return [s.to_dict() for s in signals]

    def ask(self, learner_id: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """主动提问 (频率控制自动递增)."""
        prof = self._ensure(learner_id)
        q = self._asker.next_question(learner_id, prof, context)
        if q:
            self._save(prof)  # 保存递增后的 proactive_asked
        return q

    def answer(self, learner_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """用户回答回写."""
        prof = self._ensure(learner_id)
        slot = str(payload.get("slot_key") or "")
        value = payload.get("value") or payload.get("text") or ""
        if slot in _DECLARED_BACKGROUND_SLOTS:
            text = str(value).strip()[:120]
            if text in {"跳过", "skip", "SKIP"}:
                # Remember the optional refusal so the system does not ask the
                # same question again.  The sentinel is ignored by the learner
                # prior and is never interpreted as evidence.
                prof.declared_background[slot] = "skipped"
            elif text:
                safe, _reason = self._privacy.check(text)
                if safe:
                    prof.declared_background[slot] = text
                    if slot == "learning_goal":
                        prof.goals.append({
                            "text": text,
                            "type": "short_term",
                            "confidence": 0.7,
                            "source": "declared_profile",
                        })
                        prof.goals[:] = prof.goals[-5:]
            known_declared = sum(
                1 for item in prof.declared_background.values()
                if str(item or "").strip().lower() not in {"", "skipped"}
            )
            prof.confidence = max(
                prof.confidence,
                min(0.75, known_declared / max(len(_DECLARED_BACKGROUND_SLOTS), 1)),
            )
        mapping = _SLOT_MAP.get(slot)
        if mapping:
            for field, data in mapping.items():
                if field == "goal":
                    prof.goals.append({"text": str(value)[:60], "type": data.get("type", "short_term"),
                                       "confidence": 0.7, "source": "question"})
                    prof.goals[:] = prof.goals[-5:]
                elif field == "vark" and value in ("V", "A", "R", "K"):
                    prof.vark_behavior = {k: 0.25 for k in "VARK"}
                    prof.vark_behavior[value] = 1.0
                elif field == "preference":
                    key = data.get("difficulty") if "difficulty" in data else slot
                    prof.preferences[slot] = str(value)[:40]
        if slot == "direction" and not mapping:
            prof.goals.append({"text": str(value)[:60], "type": "short_term", "confidence": 0.6, "source": "question"})
        prof.merge_from_habits()
        self._save(prof)
        return prof.to_dict()

    def correct(self, learner_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """用户纠正理解摘要."""
        prof = self._ensure(learner_id)
        field = str(payload.get("field") or "")
        value = payload.get("value")
        if field == "pace" and value in ("concentrated", "fragmented", "unknown"):
            prof.pace = value
        elif field == "expression" and value in ("concise", "detailed", "unknown"):
            prof.expression = value
        elif field == "frustration_level" and isinstance(value, (int, float)):
            prof.frustration_level = max(0.0, min(1.0, float(value)))
        prof.confidence = min(1.0, prof.confidence + 0.1)
        self._save(prof)
        return prof.to_dict()

    def clear(self, learner_id: str) -> bool:
        """清除用户画像数据 (隐私权)."""
        existed = learner_id in self._store or self.get_profile(learner_id) is not None
        self._store.pop(learner_id, None)
        service = self._profile_service
        if service is not None:
            try:
                snapshot = service.get_profile_snapshot(learner_id)
                apply_update = getattr(service, "apply_update", None)
                if snapshot is not None and callable(apply_update):
                    apply_update(
                        learner_id,
                        updates={"extras": {"user_profile": {}}},
                        expected_version=getattr(snapshot, "version", None),
                    )
            except Exception:  # noqa: BLE001 - local privacy clear still applies
                pass
        return existed

    def insights(self, learner_id: str) -> dict[str, Any]:
        """渐进式揭示摘要 (供前端注入推荐, 不含原始语料)."""
        prof = self._ensure(learner_id)
        return {
            "interests": [{"topic": i["topic"], "weight": i["weight"]} for i in prof.interests[:3]],
            "goals": [g["text"] for g in prof.goals[-2:]],
            "pace": prof.pace,
            "expression": prof.expression,
            "frustration_level": prof.frustration_level,
            "vark_behavior": dict(prof.vark_behavior),
            "confidence": prof.confidence,
        }

    # ---- 引导式咨询 (结合学情画像) ----

    def guide(self, learner_id: str, learner_snapshot: dict[str, Any] | None,
              context: dict[str, Any] | None = None) -> dict[str, Any]:
        """结合学情画像生成引导式咨询建议 (用户自己也不清楚时).

        策略:
        1. 学情薄弱点 (weak_kps / 低掌握度) 优先 — 数据驱动
        2. 兴趣命中加权: 兴趣主题命中薄弱点/任意 KP → 优先引导
        3. 无学情/无画像 → 通用引导 (建议先初测)

        Args:
            learner_snapshot: L2 学情画像 dict (kp_mastery/weak_kps/kp_names/level).
            context: 调用上下文 (utterance 等, 可选).

        Returns:
            {"direction", "reason", "suggested_kps": [{kp_id, name, mastery}],
             "next_steps": [str], "source"}
        """
        snap = learner_snapshot or {}
        mastery = snap.get("kp_mastery") or {}
        names = snap.get("kp_names") or {}
        weak = list(snap.get("weak_kps") or [])
        prof = self.get_profile(learner_id)

        # 1. 薄弱点: weak_kps 或 mastery < 0.6 的低掌握度 KP
        low_kps = []
        for kp_id, m in mastery.items():
            if float(m) < 0.6:
                low_kps.append({"kp_id": kp_id, "mastery": float(m)})
        low_kps.sort(key=lambda x: x["mastery"])
        if not weak and not low_kps:
            weak = [k["kp_id"] for k in low_kps]

        # 2. 兴趣主题 → KP 映射 (由前端/调用方通过 kp_names 提供, 此处用名称匹配)
        interest_topics = [i["topic"] for i in (prof.interests if prof else [])]
        interest_hits = []
        if interest_topics and names:
            for kp_id, m in mastery.items():
                nm = str(names.get(kp_id, kp_id))
                if any(t in nm or nm in t for t in interest_topics):
                    interest_hits.append({"kp_id": kp_id, "name": nm, "mastery": float(m)})

        # 3. 决策: 薄弱点 ∩ 兴趣 → 薄弱点 → 兴趣 → 通用
        source = "learner_snapshot"
        chosen: dict[str, Any] | None = None
        if interest_hits and low_kps:
            inter_ids = {h["kp_id"] for h in interest_hits}
            weak_ids = {k["kp_id"] for k in low_kps}
            overlap = weak_ids & inter_ids
            if overlap:
                source = "mixed"
                chosen = {
                    "kp_id": next(iter(overlap)),
                    "name": names.get(next(iter(overlap)), next(iter(overlap))),
                    "mastery": mastery.get(next(iter(overlap)), 0.0),
                }
            else:
                source = "mixed"
                chosen = {
                    "kp_id": interest_hits[0]["kp_id"],
                    "name": interest_hits[0].get("name", interest_hits[0]["kp_id"]),
                    "mastery": interest_hits[0]["mastery"],
                }
        elif low_kps:
            source = "learner_snapshot"
            chosen = {
                "kp_id": low_kps[0]["kp_id"],
                "name": names.get(low_kps[0]["kp_id"], low_kps[0]["kp_id"]),
                "mastery": low_kps[0]["mastery"],
            }
        elif interest_hits:
            source = "interest"
            chosen = {
                "kp_id": interest_hits[0]["kp_id"],
                "name": interest_hits[0].get("name", interest_hits[0]["kp_id"]),
                "mastery": interest_hits[0]["mastery"],
            }

        if chosen is None:
            # 通用引导: 无数据 → 先初测
            return {
                "direction": "initial_assessment",
                "reason": "我还不了解你的学习情况，建议先做一次基础测试，我会根据结果为你定制方向。",
                "suggested_kps": [],
                "next_steps": ["完成初测", "查看薄弱点分析", "生成今日推荐"],
                "source": "fallback",
            }

        # 生成建议文案与下一步
        name = chosen["name"]
        kp_id = chosen["kp_id"]
        mastery_val = chosen["mastery"]
        if source == "mixed" and mastery_val < 0.6:
            reason = (f"结合你的学情，{name} 是当前掌握度较低的知识点（{int(mastery_val * 100)}%），"
                      f"而且你之前对它表现出兴趣，从这里开始最容易坚持。")
        elif source == "interest":
            reason = f"你之前对 {name} 表现出兴趣，学自己感兴趣的内容效率更高。"
        else:
            reason = f"根据学情画像，{name} 是你的薄弱点（掌握度 {int(mastery_val * 100)}%），建议优先巩固。"
        next_steps = [
            f"先复习 {name} 的基础概念",
            f"做 3-5 道 {name} 相关练习",
            "完成后我会更新你的画像并推荐下一步",
        ]
        suggested = []
        for k in low_kps[:3]:
            suggested.append({"kp_id": k["kp_id"], "name": names.get(k["kp_id"], k["kp_id"]), "mastery": k["mastery"]})
        if not suggested:
            suggested = [{"kp_id": kp_id, "name": name, "mastery": mastery_val}]
        return {
            "direction": kp_id,
            "reason": reason,
            "suggested_kps": suggested,
            "next_steps": next_steps,
            "source": source,
        }

    # ---- 快照集成 ----

    def apply_to_snapshot(self, snapshot: Any, profile: UnderstandingProfile | None = None) -> None:
        """将画像写入 LearnerSnapshot.extras["user_profile"]."""
        prof = profile or self.get_profile(getattr(snapshot, "learner_id", ""))
        if prof is None:
            return
        extras = dict(getattr(snapshot, "extras", {}) or {})
        extras["user_profile"] = prof.to_dict()
        snapshot.extras = extras
