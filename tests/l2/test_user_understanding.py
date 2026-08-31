"""用户理解体系单元测试 — models / privacy / extractor / inference / distiller / asker / service."""
import pytest

from dy3_polaris.l2.user_understanding.models import (
    SignalType, UserSignal, UnderstandingProfile, HabitRecord,
)


# ============================================================
# Task 1: 数据模型
# ============================================================


def test_signal_to_dict_roundtrip():
    s = UserSignal(
        learner_id="DY20240001",
        signal_type=SignalType.INTEREST,
        payload={"topic": "浓度猝灭", "weight": 0.8},
        source="corpus",
    )
    d = s.to_dict()
    assert d["signal_type"] == "interest"
    assert d["payload"]["topic"] == "浓度猝灭"
    r = UserSignal.from_dict(d)
    assert r == s


def test_profile_defaults():
    p = UnderstandingProfile(learner_id="DY20240001")
    assert p.interests == []
    assert p.goals == []
    assert p.pace == "unknown"
    assert p.frustration_level == 0.0
    assert p.confidence == 0.0
    assert p.proactive_asked == 0


def test_profile_update_habit_accumulates():
    p = UnderstandingProfile(learner_id="DY20240001")
    p.add_habit(HabitRecord(key="pace", value="fragmented"))
    p.add_habit(HabitRecord(key="pace", value="fragmented"))
    p.add_habit(HabitRecord(key="pace", value="fragmented"))
    p.merge_from_habits()
    assert p.pace == "fragmented"
    assert p.confidence > 0.0


def test_profile_to_dict_roundtrip():
    p = UnderstandingProfile(learner_id="DY20240001")
    p.interests = [{"topic": "x", "weight": 0.5, "source": "corpus"}]
    d = p.to_dict()
    q = UnderstandingProfile.from_dict(d)
    assert q.interests == p.interests


# ============================================================
# Task 2: 隐私门
# ============================================================

from dy3_polaris.l2.user_understanding.privacy import PrivacyGate  # noqa: E402


def test_privacy_gate_filters_sensitive():
    gate = PrivacyGate()
    text = "我的身份证号是110101199001011234，血压有点高"
    ok, reason = gate.check(text)
    assert not ok
    assert "敏感" in reason


def test_privacy_gate_allows_normal():
    gate = PrivacyGate()
    ok, reason = gate.check("我想提高物理成绩，最近在学镝离子发光")
    assert ok is True


def test_privacy_gate_sanitize():
    gate = PrivacyGate()
    clean = gate.sanitize("我的电话是13800138000，想咨询量子效率")
    assert "138" not in clean


# ============================================================
# Task 3: 语料提取器
# ============================================================

from dy3_polaris.l2.user_understanding.extractor import CorpusExtractor  # noqa: E402


def _mk_extractor():
    return CorpusExtractor(known_topics=[
        "浓度猝灭", "量子效率", "热猝灭", "发光机理", "能级跃迁", "荧光粉",
        "XRD", "PL光谱", "掺杂", "镝离子",
    ])


def test_extract_interest_signal():
    ex = _mk_extractor()
    sigs = ex.extract("DY20240001", [
        {"role": "user", "text": "浓度猝灭到底是什么原理？"},
        {"role": "user", "text": "浓度猝灭怎么避免？"},
    ])
    interests = [s for s in sigs if s.signal_type.value == "interest"]
    assert interests, "应提取到兴趣信号"
    assert interests[0].payload["topic"] == "浓度猝灭"


def test_extract_frustration_signal():
    ex = _mk_extractor()
    sigs = ex.extract("DY20240001", [
        {"role": "user", "text": "这道题太难了，完全看不懂"},
        {"role": "user", "text": "又做错了，好沮丧"},
    ])
    fr = [s for s in sigs if s.signal_type.value == "frustration"]
    assert fr
    assert fr[0].payload["level"] >= 0.5


def test_extract_goal_signal():
    ex = _mk_extractor()
    sigs = ex.extract("DY20240001", [
        {"role": "user", "text": "我想考研，考材料物理方向"},
    ])
    goals = [s for s in sigs if s.signal_type.value == "goal"]
    assert goals
    assert goals[0].payload["type"] == "long_term"


def test_extract_pace_signal_from_timestamps():
    ex = _mk_extractor()
    sigs = ex.extract("DY20240001", [
        {"role": "user", "text": "看个视频", "ts": 1000},
        {"role": "user", "text": "练几道题", "ts": 1002},
        {"role": "user", "text": "再看个图", "ts": 1004},
    ])
    pace = [s for s in sigs if s.signal_type.value == "pace"]
    assert pace
    assert pace[0].payload["pace"] == "fragmented"


def test_extract_privacy_gated():
    ex = _mk_extractor()
    sigs = ex.extract("DY20240001", [
        {"role": "user", "text": "我月薪5000，身份证号110101199001011234"},
    ])
    assert sigs == []


# ============================================================
# Task 4: 画像推理引擎
# ============================================================

from dy3_polaris.l2.user_understanding.inference import ProfileInference  # noqa: E402


def test_inference_applies_signals_to_profile():
    inf = ProfileInference()
    p = UnderstandingProfile(learner_id="DY20240001")
    sigs = [
        UserSignal("DY20240001", SignalType.INTEREST, {"topic": "浓度猝灭", "weight": 0.8}, source="corpus"),
        UserSignal("DY20240001", SignalType.GOAL, {"text": "想考研", "type": "long_term"}, source="corpus"),
        UserSignal("DY20240001", SignalType.FRUSTRATION, {"level": 0.6}, source="corpus"),
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
        UserSignal("DY20240001", SignalType.VARK, {"V": 0.4, "R": 0.6}, source="corpus"),
    ]
    inf.apply(p, sigs)
    assert p.interests and p.interests[0]["topic"] == "浓度猝灭"
    assert p.goals and p.goals[0]["type"] == "long_term"
    assert p.frustration_level == pytest.approx(0.6)
    assert p.pace == "fragmented"
    assert p.vark_behavior.get("R") == pytest.approx(0.6)


def test_inference_merges_existing_interest():
    inf = ProfileInference()
    p = UnderstandingProfile(learner_id="DY20240001")
    p.interests = [{"topic": "发光机理", "weight": 0.3, "source": "corpus", "count": 1}]
    sigs = [UserSignal("DY20240001", SignalType.INTEREST, {"topic": "发光机理", "weight": 0.4, "count": 2}, source="corpus")]
    inf.apply(p, sigs)
    assert len(p.interests) == 1
    assert p.interests[0]["weight"] > 0.3  # 权重应增长


def test_inference_low_weight_no_dupe():
    inf = ProfileInference()
    p = UnderstandingProfile(learner_id="DY20240001")
    sigs = [UserSignal("DY20240001", SignalType.INTEREST, {"topic": "量子效率", "weight": 0.1}, source="corpus")]
    inf.apply(p, sigs)
    assert len(p.interests) == 1


# ============================================================
# Task 5: 记忆蒸馏器
# ============================================================

from dy3_polaris.l2.user_understanding.distiller import MemoryDistiller  # noqa: E402
from dy3_polaris.l2.user_understanding.models import SignalCategory  # noqa: E402


def test_distiller_classifies_habit_by_frequency():
    dist = MemoryDistiller(habit_threshold=3)
    sigs = [
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
        UserSignal("DY20240001", SignalType.INTEREST, {"topic": "热猝灭", "weight": 0.5}, source="corpus"),
    ]
    task, habit = dist.distill(sigs)
    assert any(s.category == SignalCategory.HABIT for s in habit)
    assert len(task) == 1  # 单次兴趣 → task
    assert any(s.category == SignalCategory.TASK for s in task)


def test_distiller_goal_always_habit():
    dist = MemoryDistiller(habit_threshold=3)
    sigs = [UserSignal("DY20240001", SignalType.GOAL, {"text": "考研", "type": "long_term"}, source="corpus")]
    task, habit = dist.distill(sigs)
    assert any(s.category == SignalCategory.HABIT for s in habit)


def test_distiller_returns_habit_records():
    dist = MemoryDistiller(habit_threshold=2)
    sigs = [
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
        UserSignal("DY20240001", SignalType.PACE, {"pace": "fragmented"}, source="corpus"),
    ]
    task, habit = dist.distill(sigs)
    recs = dist.to_habit_records(habit)
    assert any(r.key == "pace" and r.count >= 2 for r in recs)


# ============================================================
# Task 6: 澄清式提问引擎 (观察为主, 仅在模糊时提问)
# ============================================================

from dy3_polaris.l2.user_understanding.asker import ProactiveAsker  # noqa: E402


def test_asker_observation_first_no_question_when_clear():
    """意图清晰/无需澄清时不主动提问 (观察为主)."""
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")  # 冷启动也不问
    q = ask.next_question("DY20240001", p, {"view": "overview"})
    assert q is None


def test_asker_ambiguous_triggers_clarify():
    """请求难以理解/意图模糊时, 触发澄清式提问."""
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")
    q = ask.next_question("DY20240001", p, {"ambiguous": True, "intent": "query"})
    assert q is not None
    assert q["trigger"] in ("ambiguous", "clarify")
    assert "笼统" in q["question"]


def test_asker_frequency_limit():
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")
    p.proactive_asked = 3
    q = ask.next_question("DY20240001", p, {"ambiguous": True})
    assert q is None


def test_asker_skippable():
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")
    q = ask.next_question("DY20240001", p, {"ambiguous": True, "intent": "query"})
    assert q["options"] and "跳过" in q["options"]


def test_asker_practice_done_no_question():
    """练习完成不再主动盘问难度 (观察为主)."""
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")
    p.confidence = 0.9
    q = ask.next_question("DY20240001", p, {"practice_done": True})
    assert q is None


def test_asker_clarify_by_intent():
    """按意图返回对应的澄清模板."""
    ask = ProactiveAsker(max_per_session=3)
    p = UnderstandingProfile(learner_id="DY20240001")
    q = ask.next_question("DY20240001", p, {"ambiguous": True, "intent": "practice"})
    assert q is not None
    assert q["slot_key"] == "practice_scope"


# ============================================================
# Task 7: 服务门面
# ============================================================

from dy3_polaris.l2.user_understanding.service import UserUnderstandingService  # noqa: E402


def _mk_service():
    store = {}
    return UserUnderstandingService(profile_store=store)


def test_service_extract_updates_profile():
    svc = _mk_service()
    svc.extract("DY20240001", [
        {"role": "user", "text": "浓度猝灭到底是什么？我想考研"},
    ])
    prof = svc.get_profile("DY20240001")
    assert prof is not None
    assert prof.interests or prof.goals


def test_service_ask_respects_frequency():
    svc = _mk_service()
    for _ in range(5):
        q = svc.ask("DY20240001", {"ambiguous": True, "intent": "query"})
        if q is None:
            break
    prof = svc.get_profile("DY20240001")
    assert prof.proactive_asked <= 3


def test_service_ask_observation_first_no_question():
    """意图清晰时 ask 不返回问题, 也不产生画像副作用 (观察为主)."""
    svc = _mk_service()
    q = svc.ask("DY20240001", {"view": "overview"})
    assert q is None
    prof = svc.get_profile("DY20240001")
    assert prof is None or prof.proactive_asked == 0


def test_service_answer_applies_slot():
    svc = _mk_service()
    svc.answer("DY20240001", {"slot_key": "direction", "value": "准备考试", "text": "准备期末考试"})
    prof = svc.get_profile("DY20240001")
    assert prof.goals


def test_service_correct_updates_profile():
    svc = _mk_service()
    svc.correct("DY20240001", {"field": "pace", "value": "concentrated"})
    prof = svc.get_profile("DY20240001")
    assert prof.pace == "concentrated"


def test_service_clear_removes_profile():
    svc = _mk_service()
    svc.extract("DY20240001", [{"role": "user", "text": "想考研"}])
    assert svc.get_profile("DY20240001") is not None
    svc.clear("DY20240001")
    assert svc.get_profile("DY20240001") is None


def test_service_insights_returns_interpretable():
    svc = _mk_service()
    svc.extract("DY20240001", [
        {"role": "user", "text": "浓度猝灭怎么避免？"},
        {"role": "user", "text": "量子效率怎么提高？"},
    ])
    ins = svc.insights("DY20240001")
    assert isinstance(ins, dict)
    assert "interests" in ins
    assert "expression" in ins


def test_service_apply_to_snapshot_extras():
    from dy3_polaris.l2.models import LearnerSnapshot
    svc = _mk_service()
    svc.extract("DY20240001", [{"role": "user", "text": "想考研，最近在看视频学荧光粉"}])
    prof = svc.get_profile("DY20240001")
    snap = LearnerSnapshot(learner_id="DY20240001", snapshot_ts=0)
    svc.apply_to_snapshot(snap, prof)
    assert snap.extras.get("user_profile", {}).get("goals")


# ============================================================
# Task 12: 引导式咨询 (结合学情画像)
# ============================================================


def _mk_learner_snapshot(weak_kps=("kp_qe",), mastery=None, names=None):
    """构造学情画像快照 dict (模拟 L2 /l2/profile 返回结构)."""
    return {
        "learner_id": "DY20240001",
        "kp_mastery": mastery or {
            "kp_qe": 0.35, "kp_sc": 0.42, "kp_pl": 0.78, "kp_te": 0.65,
        },
        "weak_kps": list(weak_kps),
        "kp_names": names or {
            "kp_qe": "量子效率", "kp_sc": "浓度猝灭", "kp_pl": "PL光谱", "kp_te": "热猝灭",
        },
        "level": "intermediate",
        "theta": 0.4,
    }


def test_guide_returns_structured_advice():
    svc = _mk_service()
    # 用户表达迷茫: 不知道学什么
    adv = svc.guide("DY20240001", _mk_learner_snapshot(), {"utterance": "不知道学什么"})
    assert adv is not None
    assert adv.get("direction")
    assert adv.get("reason")
    assert adv.get("suggested_kps")
    assert adv.get("next_steps")
    assert adv.get("source") in ("learner_snapshot", "interest", "mixed")


def test_guide_prefers_weak_kp():
    svc = _mk_service()
    # 无兴趣画像, 仅学情: 薄弱点优先
    adv = svc.guide("DY20240001", _mk_learner_snapshot(), {})
    kp_ids = [k["kp_id"] for k in adv["suggested_kps"]]
    assert "kp_qe" in kp_ids  # 薄弱点 (mastery 0.35)
    assert adv["direction"] == "kp_qe"
    assert "量子效率" in adv["reason"]


def test_guide_interest_boost():
    svc = _mk_service()
    svc.extract("DY20240001", [{"role": "user", "text": "浓度猝灭怎么避免？"}])
    # 兴趣命中薄弱点 → 引导建议引用兴趣
    adv = svc.guide("DY20240001", _mk_learner_snapshot(), {})
    kp_ids = [k["kp_id"] for k in adv["suggested_kps"]]
    # 兴趣 "浓度猝灭" (kp_sc) 被观察到, 但 kp_sc 掌握度 0.42 也较弱
    assert adv.get("source") in ("learner_snapshot", "interest", "mixed")
    assert adv["reason"]


def test_guide_empty_snapshot_fallback():
    svc = _mk_service()
    # 无学情数据 + 无画像 → 通用引导 (不崩溃)
    adv = svc.guide("DY20240001", {"kp_mastery": {}, "weak_kps": [], "kp_names": {}}, {})
    assert adv is not None
    assert adv.get("next_steps")


def test_guide_no_weak_uses_interest():
    svc = _mk_service()
    svc.extract("DY20240001", [{"role": "user", "text": "帮我查一下 热猝灭 的知识"}])
    # 无薄弱点 (全部 mastery >= 0.6), 兴趣命中 → 引导到兴趣
    snap = _mk_learner_snapshot(weak_kps=(), mastery={
        "kp_qe": 0.75, "kp_sc": 0.80, "kp_pl": 0.78, "kp_te": 0.65,
    })
    adv = svc.guide("DY20240001", snap, {})
    kp_ids = [k["kp_id"] for k in adv["suggested_kps"]]
    assert "kp_te" in kp_ids  # 兴趣 热猝灭
    assert adv["source"] in ("interest", "mixed")
