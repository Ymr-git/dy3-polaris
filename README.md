# DY³⁺ Polaris

**面向绿色健康照明场景的稀土发光材料科研证据分析与多智能体协同决策系统**

上海云之脑智能科技有限公司「领域知识个性化生成与多智能体协同决策系统」研究比赛参赛作品（编号 XH-202630）。

系统将用户提出的材料问题转化为可追踪、可审核、可继续学习的任务：四个科研角色 Agent（学情诊断 / 知识生成 / 科学审核 / 导学决策）完成「分析—生成—校验—决策」协同闭环，输出**带来源、带边界、带不确定性说明**的建议。

一句话价值主张：**让每一个发光材料建议都能回答——依据是什么、谁质疑过、哪些仍不能确定。**

---

## 一、快速开始

### 方式 A：Windows 一键启动（推荐）

1. 安装 [Python 3.10+](https://www.python.org/downloads/)（安装时勾选 *Add Python to PATH*）；
2. 双击本目录下 **`一键启动.bat`**；
3. 脚本自动完成：探测 Python → 安装依赖 → 启动服务 → 打开浏览器；
4. 浏览器访问 <http://127.0.0.1:8000/>，用下方默认账号登录。

### 方式 B：手动安装（Windows / macOS / Linux）

```bash
# 1. 进入项目目录
cd 04-编码

# 2. 创建虚拟环境（可选但推荐）
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate

# 3. 安装依赖（含开发/测试依赖）
python -m pip install -e ".[dev]"

# 4. 创建本地配置
# Windows:  copy .env.example .env      macOS/Linux: cp .env.example .env

# 5. 启动
python -m dy3_polaris.main --host 127.0.0.1 --port 8000
```

浏览器访问 <http://127.0.0.1:8000/>。

> **无外部 LLM Key 也能完整运行**：系统内置模板+规则兜底，未配置模型时仍可完成教学骨架与证据阅读；在 `.env` 或前端设置面板配置任一受支持 provider（deepseek / anthropic / openai / qwen / zhipu / ollama / custom）后，生成质量自动升级。

---

## 二、默认账号（演示/评审用）

| 账号 | 角色 | 密码 | 画像 |
|---|---|---|---|
| DY20240001 | 本科生 | demo123 | 一无所知（差异化初始学情） |
| DY20240003 | 研究生 | demo123 | 略懂 |
| DY20240004 | 科研员 | demo123 | 熟悉 |
| DY20240002 | 教师 | demo123 | 掌握 |
| DY20248888 | 管理员 | admin888 | — |

---

## 三、验证系统已正常部署

### 3.1 健康检查

```bash
curl http://127.0.0.1:8000/health
# 期望：{"code":0,"data":{"status":"healthy","layers":{...全部 healthy}}}
```

### 3.2 知识库已加载（9313 实体 / 3074 切片 / 379 三元组）

```bash
curl http://127.0.0.1:8000/api/info
# 端点清单中应包含 /api/query 等 23 个主端点
```

### 3.3 自动化测试（约 2–5 分钟）

```bash
python -m pytest tests/test_l7_frontend.py -q      # 前端渲染契约
python -m pytest tests/l5 -q                        # 多智能体主链回归（935 项）
python -m pytest tests/contract -q                  # 竞赛就绪契约
```

### 3.4 竞赛评测重跑（84 案例，约 2 分钟）

```bash
python scripts/competition_eval.py
# 复现：84 案例、领域行为准确率 98.04%、不安全发布 0、幻觉率 0.0%
# 结果与仓库内 competition_eval_report_focus_gate_final_v2.json 一致
```

---

## 四、系统能力一览

| 能力 | 说明 |
|---|---|
| 多智能体协同 | 学情诊断 → 知识生成 → 科学审核 → 导学决策，同一 task_id 闭环 |
| 反幻觉发布门 | FactChecker 事实校验 + CC1 反幻觉校验；证据不足时诚实拒答 |
| 修订闭环 | 审核不通过触发「补充检索 → 修订生成 → 复核」，冲突全程可追踪 |
| 个性化资源 | 定制讲解 / 实操指南 / 分阶测验三种形态，带显式匹配依据 |
| 个性化画像 | BKT 知识追踪（42 KP）+ IRT 能力估计 + 四域掌握度 + 学习画像 |
| 领域知识 | 知识图谱（9313 实体 / 379 三元组）+ 混合检索（BM25+向量+图+重排） |
| 证据与溯源 | 主张-证据映射、证据链路（Chunk→Concept→Claim→Document）、X-Trace-Id 全链路 |
| 前端五视图 | 学习总览 / 任务工作区 / 协同分析 / 知识证据 / 成长路径（原生 JS SPA） |
| 治理与安全 | JWT/ABAC、安全网关、审计日志、HITL 人工介入、健康探针、幂等 |

## 五、目录结构

```
04-编码/
├── 一键启动.bat            # Windows 一键部署启动
├── pyproject.toml          # 依赖与项目元数据（pip install -e ".[dev]"）
├── README.md               # 本文件
├── .env.example            # 配置模板（复制为 .env 后填写 LLM Key）
├── src/dy3_polaris/        # 全部源代码（150 个模块，L0–L7 分层）
│   ├── main.py             # 入口：python -m dy3_polaris.main
│   ├── l0…l7/              # 治理 / 用户域 / 个性化 / 领域知识 / 决策引擎 / Agent运行时 / 协议 / 体验呈现
│   └── l7/static/          # 前端 SPA（原生 JS，零构建）
├── src/dy3_polaris/l3/data/snapshots/  # ★ 知识库快照（9313 实体/3074 切片/379 三元组，随仓库分发）
├── src/dy3_polaris/l2/data/learners/   # ★ 差异化学习者初始学情数据（演示种子）
├── tests/                  # 189 个测试文件（契约/集成/各层）
├── scripts/                # 评测与工具脚本
├── evals/                  # 评测基准数据
└── competition_eval_report_focus_gate_final_v2.json   # 84 案例评测报告（最终版）
```

## 六、配置说明（.env）

```ini
DY3_LLM_PROVIDER=deepseek        # deepseek / anthropic / openai / qwen / zhipu / ollama / custom
DY3_LLM_API_KEY=sk-你的密钥       # 留空则使用模板+规则兜底
DY3_LLM_BASE_URL=                # 可选，默认用 provider 官方端点
DY3_LLM_MODEL=                   # 可选，默认用 provider 推荐模型
DY3_LLM_TEMPERATURE=0.3
DY3_LLM_TIMEOUT=120
# NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD  # 可选外部图存储（当前主链为内存存储，无需配置）
```

密钥纪律：`.env` 已被 .gitignore 排除，**切勿提交真实 API Key**。

## 七、常见问题

| 问题 | 解决 |
|---|---|
| 一键启动提示找不到 Python | 安装 Python 3.10+ 并勾选 Add Python to PATH |
| 依赖安装失败 | 网络受限时改用镜像：`pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 问答耗时较长 | 未配置 LLM Key 时走模板兜底（较快）；配置 Key 后走真实模型（首次调用需连接） |
| 前端页面空白 | 硬刷新（Ctrl+F5）；确认启动日志无异常 |
| 想恢复初始演示数据 | 删除运行时生成的 `data/` 目录后重启（知识库快照随仓库保留） |

## 八、边界声明（诚实工程）

- 主链以**内存存储**运行；Neo4j/Milvus/PostgreSQL 等为可选适配器与设计，未用于支撑主链；
- 四 Agent 为同进程同步编排（同一 task_id），非独立进程自主辩论；
- 前端为原生 JavaScript（设计文档中的 React 方案未采用，换取零构建开箱即用）；
- 健康照明标准（CCT/CRI/蓝光）为教育演示口径，不构成权威合规结论；
- 系统定位为「辅助分析与可信决策」，不替代实验与专家判断。

---

*更多信息见参赛材料：作品设计与实现方案（DOCX，1.5 万字+）、竞赛 PPT（37 页）、设计图目标规格（OCR 提取）。*
