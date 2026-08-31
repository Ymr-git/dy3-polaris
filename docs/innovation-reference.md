# 可借鉴方案调研（创新点参考）

> 生成时间: 2026-08-14
> 目的: 调研多智能体协同决策 / 个性化学习 / RAG 幻觉防御领域的成熟与前沿（含争议）方案，
>       映射到 DY3-Polaris 现有架构，作为创新点来源。

## 一、多智能体教育系统（成熟 + 前沿）

| 方案 | 核心思想 | 对本项目的借鉴点 |
|---|---|---|
| **LEA** (Learning Engagement Assistant) | 用「模拟学生代理」做自适应个性化学习 | 当前用 10 个静态画像模拟，可升级为「学生代理」动态仿真反馈 |
| **Learning in Blocks**（AIED 2025） | 多代理辩论辅助个性化自适应学习 | 与「多智能体协同决策」定位一致；可借鉴「辩论驱动个性化路径」 |
| **Adaptive Multi-Agent Tutoring** | 自适应多代理辅导 | 诊断→生成→审核→决策四 Agent 已有，可补「实时认知/心理状态评估」维度 |

来源：[LEA](https://rke.abertay.ac.uk/en/publications/learning-engagement-assistant-lea-a-multi-agent-ai-framework-for-/)、[Learning in Blocks](https://dl.acm.org/doi/10.1007/978-3-032-29755-6_27#1)、[Adaptive Multi-Agent Tutoring](https://ieeexplore.ieee.org/document/11479250#1)

## 二、RAG 幻觉防御（前沿 + 争议）

| 方案 | 核心思想 | 对本项目的借鉴点 |
|---|---|---|
| **Debate-Augmented RAG**（ACL 2025） | 用「辩论」增强 RAG 消除幻觉 | 项目已有 `_run_candidate_debate`，可借鉴其更成熟的辩论机制（多轮、证据交换、胜负判定） |
| **Multi-Round Agentic RAG**（ICML 2026 "From Conflict to Consensus"） | 多轮代理 RAG 从冲突收敛到共识 | 与「多候选交叉验证 + 共识判定」一致；可借鉴多轮收敛而非一次辩论 |
| **Self-Reflective Debates for Context Reliability**（2025） | 自反思辩论评估「上下文可靠性」 | 可借鉴「上下文可靠性」评估，改进当前相关性门槛（overlap 启发式） |
| **MEGA-RAG**（Frontiers 2025） | 多证据引导答案精炼（SEAE 评分 + DISC 模块解决事实不一致） | 可借鉴「多证据精炼」改进答案合成 |

来源：[Debate-Augmented RAG](https://aclanthology.org/2025.acl-long.770/)、[Multi-Round Agentic RAG](https://icml.cc/virtual/2026/poster/63972)、[Self-Reflective Debates](https://huggingface.co/papers/2506.06020)、[MEGA-RAG](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1635381/full)

## 三、与现有架构的映射与创新方向

当前已有（可对标）：多候选生成 + 交叉验证 + 协同辩论（`_run_candidate_debate`）、8 维度验证、BKT/IRT 画像、知识图谱分层。

**可落地的创新点**（按性价比）：
1. **辩论升级**：当前辩论是「一轮、二分」；借鉴 Debate-Augmented RAG / Multi-Round Agentic RAG，升级为「多轮辩论收敛 + 上下文可靠性自反思」，直接增强现有 `_run_candidate_debate`。
2. **上下文可靠性评分**：借鉴 Self-Reflective Debates，给每条检索证据打「可靠性分」，替代当前粗粒度 overlap 启发式。
3. **模拟学生代理**：借鉴 LEA，用代理动态仿真「被动/主动」画像，替代静态画像脚本（画像循环可自动化）。
4. **实时认知状态评估**：借鉴 Adaptive Multi-Agent Tutoring，在学情诊断 Agent 里加「认知/心理状态」维度。

*注：上述为调研线索，具体实现需结合当前代码架构（L4/L5）评估后落地。*
