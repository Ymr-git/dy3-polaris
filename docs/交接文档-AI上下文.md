# DY3-Polaris 项目交接文档（供新会话 AI 快速上手）

## 一、项目是什么

**DY3-Polaris** —— 面向「稀土发光材料」领域的智能教学与问答系统（多智能体协同决策）。

- 定位：竞赛项目（上海云之脑公司「领域知识个性化生成与多智能体协同决策系统」比赛方案）。
- 核心能力：多智能体协同（学情诊断→知识生成→审核校验→导学决策）、个性化画像（BKT/IRT）、知识图谱、练习出题、知识问答、个性化资源生成。
- **不依赖外部 LLM 也能跑**（模板+规则兜底），配 API key 后走 LLM 重组。

### 目录结构

```
D:\BaiduNetdiskDownload\xiaotiao\dy3-polaris-整理后\
├── 01-规划\        # 架构总览、路线图、竞赛计划、部署架构
├── 02-设计\        # L0-L7 各层设计 + CC1-CC3 + 专项设计（HTML）
├── 03-信息文档\     # 交接文档、项目索引、报告
├── 04-编码\        # ★ 核心代码
│   ├── src\dy3_polaris\
│   │   ├── l0/ 治理层   l1/ 用户域   l2/ 个性化   l3/ 领域知识
│   │   ├── l4/ 决策引擎 l5/ Agent运行时 l6/ 协议   l7/ 体验呈现(含前端)
│   │   └── main.py 入口
│   ├── src\dy3_polaris\l7\static\   # ★ 前端（原生 JS，非 React）
│   │   ├── index.html
│   │   └── assets\  app.js(路由内核) mf6-features.js(20个视图) mf7-assistant.js
│   │                  mf8-atomic-viz.js mf10-kg-viz.js(知识图谱) dp-collab.js
│   ├── docs\persona-library.md   # ★ 用户画像库 + 迭代记录
│   ├── tests\        # 测试
│   └── pyproject.toml
├── 05-验证\        # 验证报告
├── 一键启动.bat     # 双击启动（自动探测 Python + 装依赖 + 开浏览器）
└── 启动教程-小白版.docx / .txt
```

## 二、技术栈与启动

- 后端：Python 3.14 + Starlette + uvicorn，L0-L7 八层架构（内存存储为主）。
- 前端：**原生 JS**（不是 React，尽管设计书写 React）——`l7/static/index.html` + `assets/*.js`。
- 数据：L1 用户/鉴权、L2 画像(BKT知识追踪+IRT能力+掌握度)、L3 知识库(实体+三元组+混合检索)、L4 决策引擎(next-action)、L5 Agent运行时(交互记录器)。
- 启动：`cd 04-编码` 后 `python -m dy3_polaris.main --port 8000`，访问 http://localhost:8000/
- 默认账号：DY20240001(本科生)/demo123、DY20240002(教师)/demo123、DY20240003(研究生)/demo123、DY20240004(科研员)/demo123、DY20248888(管理员)/admin888
- Python 环境：`整理后\.venv\Scripts\python.exe`（依赖已装好）

## 三、当前工作：画像测试 + 功能优化（持续迭代任务）

用户要求：设计多维度画像（身份×能力×方向 + 主动/被动），逐个带需求体验系统 → 发现不足 → 改进 → 再体验，**反复循环**。已跑 3 轮 + 扩展功能测试。

**画像库**：`04-编码/docs/persona-library.md`（8 个画像 + 完整迭代记录，是接续工作的核心依据）。

**测试方法**：写 Python 脚本用 httpx 通过 HTTP API 模拟画像体验（登录→画像→推荐→练习→问答→图谱→个性化资源），运行后分析系统返回，发现不足。

## 四、本会话累计完成的关键改动（16 个）

### 功能新增
1. 知识图谱 L1-L4 分层（后端 seed 层级实体 + 前端 mf10-kg-viz.js 分层径向布局）
2. 个性化学习资源 3 形态（`/api/personalized/resources`：定制讲解+实操指南+分阶测试题）
3. 4 档能力画像账号（DY20240001/03/04/02 = 一无所知/略懂/熟悉/掌握）
4. LLM 配置现代化（provider 预设 + `/api/llm/config` 端点 + 前端只填 key）

### Bug 修复（问答质量）
5. 意图识别：method 优先于 definition；`怎样(?!的)` 排除"怎样的"
6. 幻觉防御 `_ghost` 误触发 → 加连接词过滤
7. 口语词稀释检索 → `normalize_query` 加口语词过滤
8. 检索不相关 → `run_generation` 加主题词过滤
9. 答案格式：清理 Markdown 标题/编号/`<sup>`标签/乱码`ꎬ`/行中编号/标题图片噪声
10. kp_name 编号→中文名（`kp_catalog.kp_name()`）
11. BKT 双重更新 → `skip_bkt_update` 参数
12. 薄弱点阈值统一 0.5→0.6
13. 雷达图 `k.id`→`k.code`、推荐目标 NaN 兜底
14. WS 通道"已有登录态"不连接 → 加载时立即连接
15. LLM 配置前端自动同步到后端
16. PPT 讲解：`renderSlideDeck` 结构化幻灯片（标题+内容）

## 五、待办 / 遗留问题

1. **检索不相关残留**：主题词过滤已修，但混合检索对近义词（机理/光谱/猝灭）区分仍可能不够精准，可继续优化 reranker 或检索参数。
2. **个性化资源 role 映射**：科研员(GRADUATE role)显示"研究生"，应综合 L2 level 显示。
3. **数值/化学式丢失**：PDF 转文本 `<sub>/<sup>` 数值丢失（"Al O"应为 Al₂O₃），数据质量问题，靠 LLM 缓解。
4. **规划书缺失功能**（验收清单里发现的硬缺口）：
   - 实验导学系统（8 步流程 + 苏格拉底追问）——规划书写"特色功能"但代码没有
   - 职业方向推荐
   - 五维能力雷达（实际四维）、前端 React（实际原生 JS）
   - 知识库 44 条（DOMAIN_KNOWLEDGE 已清空，用 wxk 文献替代）
   - 量化指标（幻觉率<5%、适配准确率≥85%、覆盖率≥90%）无量化数据

## 六、如何继续

1. 重启后端（`.venv\Scripts\python.exe -m dy3_polaris.main --port 8000`）+ 浏览器硬刷新。
2. 从 `04-编码/docs/persona-library.md` 的「迭代记录」接上进度。
3. 继续画像测试循环，或按验收清单补缺失功能。
4. 测试脚本是临时的（用完即删），画像库是持久的（记录所有发现和修复）。

## 七、关键代码位置速查

- 问答链路：`l5/unified_app.py`(api_query) → `l5/agent_workers.py`(run_generation 检索+合成+幻觉防御)
- 画像：`l2/profile_builder/`(builder/tracing_service) + `l2/kp_catalog.py`(42 KP 目录)
- 知识库：`l3/`(store/retrieval/knowledge_seed)
- 个性化资源：`l5/unified_app.py` 的 `api_personalized_resources` + `_build_customized/_build_practical_guide/_build_staged_questions`
- 前端视图：`l7/static/assets/mf6-features.js`(renderQuery/renderPractice/renderWeakPoints/renderLearnerOverview)
- 知识图谱：`l7/static/assets/mf10-kg-viz.js` + `l3/api/router.py` 的 graph_hierarchy
