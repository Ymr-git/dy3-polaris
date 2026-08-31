"""L2 知识点职业角色关联 — 镝-绿色健康照明「多职业」维度.

在 42 KP 之上叠加「职业角色」层: 不同职业角色关心的知识点子集不同,
用于支持按角色的个性化学习路径推荐 (对标 KnowEdu 的 prerequisite 教学策略
+ K12-KGraph 的课程对齐思想)。

数据:
  - ROLES            : 7 种职业角色 (学生/教师/材料工程师/照明设计师/研究员/健康专家/质量工程师)
  - ROLE_KP          : 角色 → {kp_id: 权重} (1.0 核心 / 0.6 相关)
  - HEALTH_LIGHTING_KPS : 「健康照明应用」跨域维度标签 (跨 A/B/D 域, 突出主题主线)
"""
from __future__ import annotations

from typing import Any

# 职业角色 (key 为稳定英文 id, name 为中文)
ROLES: dict[str, dict[str, str]] = {
    "learner": {"name": "学生", "desc": "系统学习者，需建立从机理到应用的全面基础"},
    "teacher": {"name": "教师", "desc": "教学者，关注知识点前后置关系与教学策略"},
    "materials_engineer": {"name": "材料工程师", "desc": "研发者，关注基质设计、掺杂与合成工艺"},
    "lighting_designer": {"name": "照明设计师", "desc": "应用者，关注色温、显色、蓝光危害与场景"},
    "researcher": {"name": "研究员", "desc": "学术研究者，关注发光机理、能级与理论"},
    "health_expert": {"name": "健康专家", "desc": "健康领域，关注蓝光危害与光生物效应"},
    "quality_engineer": {"name": "质量工程师", "desc": "生产者，关注表征、批次一致性与标准"},
}

# 角色 → 核心/相关 KP (权重 1.0=核心, 0.6=相关)
ROLE_KP: dict[str, dict[str, float]] = {
    "learner": {
        "A-01": 1.0, "A-03": 1.0, "A-04": 1.0, "A-05": 1.0,
        "A-11": 0.6, "A-12": 1.0, "A-13": 0.6,
        "B-01": 0.6, "B-02": 0.6, "B-05": 0.6, "B-06": 0.6, "B-07": 1.0,
        "C-01": 1.0, "D-01": 1.0, "D-03": 1.0,
    },
    "teacher": {},  # 教师全览: 运行时由 kp_catalog.ALL_KP_IDS 补齐
    "materials_engineer": {
        "B-01": 1.0, "B-02": 1.0, "B-03": 1.0, "B-04": 1.0, "B-05": 1.0,
        "B-08": 0.6, "B-09": 1.0, "B-10": 0.6,
        "C-01": 1.0, "C-02": 1.0, "C-03": 1.0, "C-04": 1.0,
        "C-05": 1.0, "C-06": 1.0, "C-07": 1.0, "C-08": 1.0, "C-09": 1.0, "C-10": 1.0,
    },
    "lighting_designer": {
        "A-05": 0.6, "B-06": 0.6, "B-07": 1.0, "B-08": 1.0, "D-08": 1.0,
    },
    "researcher": {
        "A-01": 0.6, "A-02": 0.6, "A-03": 1.0, "A-04": 1.0, "A-05": 1.0,
        "A-06": 1.0, "A-07": 1.0, "A-08": 1.0, "A-09": 1.0, "A-10": 1.0,
        "A-11": 1.0, "A-12": 1.0, "A-13": 1.0,
        "B-05": 0.6, "B-06": 0.6,
    },
    "health_expert": {
        "A-12": 0.6, "A-13": 0.6, "B-06": 0.6, "B-07": 1.0,
    },
    "quality_engineer": {
        "C-09": 1.0, "C-10": 1.0,
        "D-01": 1.0, "D-02": 1.0, "D-03": 1.0, "D-04": 1.0,
        "D-05": 1.0, "D-06": 1.0, "D-07": 1.0, "D-08": 1.0,
    },
}

# 「健康照明应用」跨域维度标签 (跨 A/B/D 域, 突出主题主线「绿色健康照明」)
HEALTH_LIGHTING_KPS: list[str] = [
    "A-05", "A-12", "A-13",  # 发光本质 + 猝灭 (影响效率/稳定)
    "B-06", "B-07", "B-08",  # 效率 + 色坐标 + 激发 (健康照明核心参数)
    "D-08",                    # CIE 色度测量
]


def _teacher_kps() -> dict[str, float]:
    """教师全览: 全部 42 KP 权重 1.0 (惰性填充, 避免循环导入)."""
    from dy3_polaris.l2.kp_catalog import ALL_KP_IDS

    return {kp: 1.0 for kp in ALL_KP_IDS}


def role_list() -> list[dict[str, Any]]:
    """返回角色列表 (含各自核心 KP 数)."""
    out: list[dict[str, Any]] = []
    for rid, meta in ROLES.items():
        kps = ROLE_KP.get(rid) or _teacher_kps() if rid == "teacher" else (ROLE_KP.get(rid) or {})
        out.append({
            "role_id": rid,
            "name": meta["name"],
            "desc": meta["desc"],
            "kp_count": len(kps),
        })
    return out


def role_kps(role_id: str) -> dict[str, float]:
    """返回角色的 KP→权重 (teacher 返回全 42 KP)."""
    if role_id == "teacher":
        return _teacher_kps()
    return dict(ROLE_KP.get(role_id) or {})


def kp_roles(kp_id: str) -> list[dict[str, Any]]:
    """返回关注某 KP 的角色列表 (反向查询)."""
    out: list[dict[str, Any]] = []
    for rid, meta in ROLES.items():
        kps = _teacher_kps() if rid == "teacher" else (ROLE_KP.get(rid) or {})
        w = kps.get(kp_id)
        if w:
            out.append({"role_id": rid, "name": meta["name"], "weight": w})
    return out


__all__ = [
    "ROLES",
    "ROLE_KP",
    "HEALTH_LIGHTING_KPS",
    "role_list",
    "role_kps",
    "kp_roles",
]
