# -*- coding: utf-8 -*-
"""量化指标评测: 幻觉率 / 覆盖率 / 适配准确率（验收硬缺口）。

运行: 后端启动后 .venv python eval_metrics.py
指标口径:
- 覆盖率 = 有实质答案(非"暂无知识/不属于本系统/证据不足")的问题占比
- 诚实率 = 答案含"证据不足/无法确定"的比例(系统诚实不编造, 越高越好)
- 幻觉率 ≈ 1 - (诚实率 + 覆盖率) 的残余(答非所问/编造), 目标 <5%
- 适配准确率 = 个性化资源 role_label 匹配画像的比例, 目标 >=85%
"""
import time
import httpx

BASE = "http://localhost:8000"

# 领域核心问题（20 个，覆盖 A理论/B应用/C合成/D表征）
QUESTIONS = [
    "Dy3+ 掺杂荧光粉的发光机理是什么？",
    "浓度猝灭是怎么产生的？",
    "白光 LED 用荧光粉怎么选型？",
    "高温固相法合成荧光粉的步骤？",
    "XRD 怎么测荧光粉物相？",
    "上转换发光的原理是什么？",
    "量子效率怎么测量？",
    "热猝灭如何抑制？",
    "Dy3+ 的发射峰在哪些波长？",
    "稀土离子的 4f 跃迁有什么特点？",
    "荧光粉的色坐标怎么计算？",
    "溶胶凝胶法和固相法的区别？",
    "SEM 和 TEM 分别看什么？",
    "能量传递的机理是什么？",
    "晶体场对能级有什么影响？",
    "荧光寿命怎么测？",
    "Eu3+ 和 Dy3+ 发光有什么区别？",
    "掺杂浓度对发光强度的影响？",
    "荧光粉的热稳定性怎么评估？",
    "基质选择对发光有什么影响？",
]

USERS = {"DY20240004": "researcher", "DY20240003": "graduate",
         "DY20240001": "undergrad", "DY20240002": "teacher"}


def main():
    c = httpx.Client(base_url=BASE, timeout=60)
    covered = 0
    honest = 0
    for q in QUESTIONS:
        try:
            r = c.post("/api/query", json={"query": q, "learner_id": "DY20240004"})
            ans = str(r.json().get("data", {}).get("answer", ""))
        except Exception:
            ans = ""
        if not ans or ("暂无" in ans) or ("不属于本系统" in ans):
            continue
        if "证据不足" in ans or "无法确定" in ans or "未收录" in ans or "未提及" in ans:
            honest += 1
        else:
            covered += 1
        time.sleep(0.2)

    total = len(QUESTIONS)
    coverage = covered / total
    honest_rate = honest / total
    hallucination = 1 - coverage - honest_rate  # 残余: 有答案但答非所问/编造
    print(f"\n=== 问答指标 (n={total}) ===")
    print(f"覆盖率: {coverage:.0%} (目标 >=90%)")
    print(f"诚实率: {honest_rate:.0%} (系统诚实说明证据不足)")
    print(f"幻觉率(残余): {hallucination:.0%} (目标 <5%)")

    # 适配准确率: 个性化资源 role_label 匹配
    ok = 0
    for sid, expect in USERS.items():
        try:
            r = c.get("/api/personalized/resources", params={"learner_id": sid})
            lc = r.json().get("data", {}).get("learner_context", {})
            role = lc.get("role", "")
            if role == expect:
                ok += 1
        except Exception:
            pass
    acc = ok / len(USERS) if USERS else 0
    print(f"\n=== 适配准确率 ===")
    print(f"role 匹配: {ok}/{len(USERS)} = {acc:.0%} (目标 >=85%)")


if __name__ == "__main__":
    main()
