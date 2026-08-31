# 能力契约清单（Ability Contracts）

> 生成时间：2026-08-16
> 用途：把「系统该具备什么能力」写成**可执行的契约**（输入 → 期望行为 → 确定性判定），
> 作为系统改动的验收守卫。收编自三处现成黄金集，不新发明：
> 1. `scripts/competition_eval.py` — 三项硬指标（幻觉率 / 难度适配 / 覆盖率）
> 2. `docs/persona-enrichment-round4.md` — 10 画像 × QA 问题类型矩阵 + learner model
> 3. `docs/persona-library.md` 迭代记录 — 两条反复出现的待修缺口
>
> **判定原则：零 LLM。** 每条契约的「判定」都是确定性规则（正则 / 关键词 / 字段），
> 保证可复现、零成本、可进 CI。LLM 只用于「运行时增强」（意图兜底 / 语义 critic），不用于「判定对错」。

---

## 维度总览

| 维度 | 对应竞赛支柱 | 契约数 | 现状 |
|---|---|---|---|
| A. 诚实（防幻觉） | 诚实 | 2 | 1 绿 / 1 待验证 |
| B. 交互（意图识别 + 澄清） | 交互 | 3 | 全绿 |
| C. 画像适配 | 能力覆盖 + 交互 | 3 | 全绿 |
| D. 能力覆盖（知识点） | 能力覆盖 | 1 | 绿 |
| E. 多智能体 + 资源形态 | 能力覆盖 | 2 | 待验证 |

> 状态图例：🟢 已满足（契约测试必须绿）｜🔴 待修缺口（契约测试标记 xfail，修好后转绿）｜🟡 待验证（需全栈链路确认）

---

## A. 诚实（防幻觉）

### A1 · 领域外诚实拒答 🟡
- **输入**：`今天天气如何`
- **期望**：不编造，明确「不属本系统 / 暂无相关知识」
- **判定**：`knowledge_unavailable == True` 或 answer 含 `不属于本系统 / 暂无 / 无法给出 / 无法回答 / 未收录` 之一
- **来源**：persona-library #29「光怎么变颜色」被误拒的反面 —— 领域外应拒，但领域内不能误拒

### A2 · 问 A 离子不答 B 离子（张冠李戴）🟡
- **输入**：`Eu3+ 的发光机理是什么`
- **期望**：answer 以 Eu 为主体，不把 Dy 当主导
- **判定**：`_extract_ions(query) = Q` 非空时，`_count_ions(answer)` 中「非 Q 离子」的最大次数 ≤「Q 离子」的最大次数
- **来源**：persona-library #17（离子主次判断，已实现 `_count_ions`）；competition_eval 幻觉率口径的补充

### 硬指标锚（已存在，引用 competition_eval）
- **幻觉率 < 5%**：10 个标准 query 走 `/api/query`，`review.verdict ∈ {rejected, needs_review}` 占比 < 5%（口径见 `evaluate_hallucination_rate`）

---

## B. 交互（意图识别 + 澄清）

> 这三条是 2026-08-16 意图层改造（统一 IntentClassifier + LLM 兜底）的成果固化。

### B1 · 定义意图（含变体）不澄清 🟢
- **输入**：`dy是什么` / `什么是镝` / `dy是何方神圣` / `镝是啥` / `dy啥意思`
- **期望**：不触发澄清，直接作答
- **判定**：`_detect_ambiguity(q) is None`
- **来源**：本轮改造；对应「换个词也认识」的泛化诉求

### B2 · 方法 / 原因 / 比较 / 数值意图不澄清 🟢
- **输入**：`dy怎么制备` / `dy为什么发光` / `Dy和Eu的区别` / `4F9/2跃迁波长是多少`
- **期望**：不澄清
- **判定**：`_detect_ambiguity(q) is None`
- **来源**：本轮改造（IntentClassifier 补 method/reason 意图）

### B3 · 纯元素 / 过短澄清 🟢
- **输入**：`dy` / `镝` / `er` / `嗯`
- **期望**：触发澄清（信息不足，引导补全）
- **判定**：`_detect_ambiguity(q) is not None`
- **来源**：本轮改造（保留「纯元素/过短」作为澄清兜底）

---

## C. 画像适配

### C1 · 难度适配 ≥ 85% 🟢（硬指标锚）
- 见 `evaluate_difficulty_adaptation`（自适应出题 + 判题正确率）
- 来源：competition_eval

### C2 · 通俗化按画像自动降级 🟢（已修复）
- **输入**：learner=beginner（画像 1 小白）+ `发光材料是什么`
- **期望**：答案通俗（生活化比喻 + 术语带注释），**不堆学术术语**
- **判定**：answer 中学术术语（`4f / 能级 / 跃迁 / 电荷迁移 / 谱线 / 基质 / 猝灭 / 量子效率 / 激发态 / 电子构型`）出现次数 ≤ 2
- **来源**：persona-library #29④、#30④；persona-enrichment-round4 第四节「认知粒度需求」
- **修复**：`run_generation` 对 beginner 画像 + 定义式基础概念自动走 `_match_plain_concept`（不再只靠「大白话」关键词）；通俗讲解标记 `plain_language=True` 后在 `_run_multi_candidate_generation` / `run_guidance` 短路，跳过 critic 审核与自纠回路，避免被「改写 query 重新生成」覆盖成学术答案

### C3 · 检索相关性（不答非所问）🟢（Dy 跃迁 case 已修复）
- **输入**：`Dy3+ 的蓝光和黄光分别来自哪个能级跃迁`
- **期望**：answer 命中 Dy 跃迁（4F9/2 / 6H15/2 / 6H13/2 或 480/575 nm），**不串到**激子 S 态 / FS5 仪器 / 红外余辉
- **判定**：`("4F9/2" in answer 或 "480" in answer 或 "575" in answer) 且 "激子" not in answer 且 "FS5" not in answer`
- **来源**：persona-library #29③、#30②、#31、#32 遗留（最痛点）
- **⚠ 注意**：检索不相关是「类问题」，本条只锁「Dy 跃迁」一个 case（契约测试首跑即 XPASS 证明已修）；其余 case（降蓝光危害→FS5、XRD 步骤→图目录、上转换量子产率→LED 芯片）仍待补契约逐一锁定

---

## D. 能力覆盖（知识点）

### D1 · 覆盖率 ≥ 90% 🟢（硬指标锚）
- 42 KP 中被 `_KG_NODES` 映射覆盖的比例 ≥ 90%
- 见 `evaluate_coverage`（实测 100%）
- 来源：competition_eval

---

## E. 多智能体 + 资源形态

### E1 · ≥ 3 agents 形成闭环 🟡
- **期望**：一个 `/api/query` 的响应里，`loop_trace` / `candidates` / `review` 等字段证明「分析→生成→校验→决策」多 agent 协作
- **判定**：`review` 字段存在 且（`loop_trace` 非空 或 `candidates` 非空 或 `reasoning_loop` 存在）
- **来源**：竞赛方案「≥3 agents 形成 loop」

### E2 · ≥ 3 资源形态 🟡
- **输入**：`run_generation` mode ∈ {lecture, guide, practice}
- **期望**：讲义 / 实操指南 / 分阶练习三种形态都能产出非空内容
- **判定**：三种 mode 调用返回的 answer 均非空
- **来源**：竞赛方案「≥3 resource forms」；persona-library #174/#175（实验导学、职业推荐等资源）

---

## 落地方式

- 已满足（🟢/🟡 验证后）→ 普通测试，必须绿，防回归
- 待修（🔴）→ `@pytest.mark.xfail(reason="待修缺口")`，把缺口从 md 备忘变成「可跑的显式红」，修好后移除 xfail
- 本清单是「守」的规范；「攻」（发现新缺口）靠 persona 画像 × QA 问题类型矩阵继续滚
