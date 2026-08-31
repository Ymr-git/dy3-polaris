"""语义评测层 — 用 flash LLM 判「系统答案 vs 期望要点」，暴露深层硬骨头.

区别于 competition_eval 的三项硬指标（幻觉率/覆盖率/适配，可确定性判定），
语义评测针对「答到点子上吗、缺了什么」这类没有确定性规则、只能语义判分的问题。

运行：python scripts/semantic_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# golden set：query + 期望要点（每条要点都要判断「答到/未答到」）
_GOLDEN = [
    {
        "query": "PL 光谱怎么测？激发波长和狭缝怎么设",
        "points": [
            "给出 PL 光谱的测量步骤（选激发波长、设狭缝、扫描发射谱）",
            "给出具体参数值（激发波长、狭缝宽度）或说明「取决于具体样品」",
        ],
    },
    {
        "query": "如何提高上转换量子产率",
        "points": [
            "给出提高方法（核壳结构、尺寸调控、基质选择、表面钝化等）",
            "不只给机理或数量级，要给「怎么做」",
        ],
    },
    {
        "query": "如何降低白光 LED 的蓝光危害",
        "points": [
            "给出降低方法（降色温、增补红光/黄光、滤光/扩散层）",
        ],
    },
]


def _query(client, q: str) -> str:
    r = client.post("/api/query", json={"query": q, "learner_id": "DY20240001"})
    return str((r.json().get("data") or {}).get("answer") or "")


def _llm_judge(query: str, answer: str, points: list[str]) -> str:
    from dy3_polaris.l3.llm_config import chat_completion

    pts = "\n".join(f"- {p}" for p in points)
    prompt = (
        "你是领域评测专家。判断下面的系统答案是否答到了「期望要点」。\n\n"
        f"问题：{query}\n\n"
        f"期望要点：\n{pts}\n\n"
        f"系统答案：{answer}\n\n"
        "请逐条判断每条期望要点「已答到」还是「未答到」，"
        "然后给整体评分（0-10）和一句话说明「缺了什么」。"
    )
    return chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=300, disable_thinking=True,
    )


def main() -> int:
    from starlette.testclient import TestClient
    from dy3_polaris.l5.unified_app import UnifiedApp

    print("构建客户端…")
    builder = UnifiedApp.create_full_app_builder()
    client = TestClient(builder.create_app())
    print(f"评测 {len(_GOLDEN)} 个 case（flash LLM 语义判分）\n")
    for g in _GOLDEN:
        answer = _query(client, g["query"])
        print("=" * 72)
        print("Q:", g["query"])
        print("系统答案:", answer[:220])
        print("-" * 72)
        verdict = _llm_judge(g["query"], answer, g["points"])
        print("LLM 判分:", verdict)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
