"""L4 学习策略层 — 唯一策略决策点 (next-action).

策略模式:
- review: 复习模式 — 针对将遗忘/低于复习阈值的知识点生成复习建议
- guide : 导学模式 — 基于薄弱点生成练习路径 (攻克 → 巩固 → 考核)
- default: 默认模式 — 画像健康时返回继续学习建议

统一决策语义 (三合一):
- action_type : 行动类型 (review / practice / assess / learn / confirm)
- confidence  : 决策置信度
- recommended_path : 推荐 KP 步骤列表 [{kp_id, action, target, effort}]
"""
from __future__ import annotations

from typing import Any

# 薄弱阈值 (与 L2 ProfileBuilder 一致)
_WEAK_THRESHOLD = 0.6
# 复习阈值 (遗忘风险)
_REVIEW_THRESHOLD = 0.45
# 单路径最大步骤
_MAX_STEPS = 5


def _effort_of(mastery: float) -> str:
    """按当前掌握度估算投入档位."""
    if mastery >= 0.45:
        return "低"
    if mastery >= 0.3:
        return "中"
    return "高"


def _steps_from_weak(weak: list[tuple[str, float]], mode: str) -> list[dict[str, Any]]:
    """从薄弱点生成步骤 (练习/考核交替)."""
    steps = []
    for i, (kp, m) in enumerate(weak):
        action = "练习"
        if mode == "guide":
            action = "练习" if i % 2 == 0 else "考核"
        elif mode == "review":
            action = "复习"
        elif mode == "assess":
            action = "考核"
        steps.append({
            "step": i + 1,
            "kp_id": kp,
            "action": action,
            "target": round(min(0.85, m + 0.25), 2),
            "effort": _effort_of(m),
        })
    return steps


def generate_next_action(
    learner_profile: dict[str, Any] | None,
    mode: str = "default",
) -> dict[str, Any]:
    """生成下一次学习行动 (唯一策略决策).

    Args:
        learner_profile: 学习者画像 (来自 L2, 含 kp_mastery/weak_kps/level).
        mode: 策略模式 (default/review/guide/assess).

    Returns:
        统一决策体:
        {
            action_type: str,
            confidence: float,
            recommended_path: list[{kp_id, action, target, effort}],
            plan_id: str,
            mode: str,
            summary: str,
        }
    """
    profile = learner_profile or {}
    kp_mastery = profile.get("kp_mastery") or {}
    weak_kps = profile.get("weak_kps") or []

    # 薄弱点: weak_kps 优先, 缺失时按掌握度 < 阈值推导
    if weak_kps:
        weak = [(k, float(kp_mastery.get(k, 0.0))) for k in weak_kps]
    else:
        weak = [(k, m) for k, m in kp_mastery.items() if m < _WEAK_THRESHOLD]
    weak.sort(key=lambda kv: kv[1])
    weak = weak[:_MAX_STEPS]

    if not weak:
        return {
            "action_type": "learn",
            "confidence": 0.8,
            "recommended_path": [],
            "plan_id": "",
            "mode": mode,
            "summary": "当前画像无薄弱点, 建议继续学习新知识点或进行进阶考核",
        }

    action_type = "practice"
    if mode == "review":
        action_type = "review"
    elif mode == "assess":
        action_type = "assess"
    elif mode == "guide":
        action_type = "practice"

    # 置信度: 薄弱程度越明显置信度越高
    avg_weak = sum(m for _, m in weak) / len(weak)
    confidence = round(min(0.95, max(0.55, 1.0 - avg_weak)), 2)

    return {
        "action_type": action_type,
        "confidence": confidence,
        "recommended_path": _steps_from_weak(weak, mode),
        "plan_id": "",
        "mode": mode,
        "summary": (
            f"检测到 {len(weak)} 个薄弱知识点, 建议按路径依次"
            f"{'复习' if mode == 'review' else '练习/考核'}"
        ),
    }
