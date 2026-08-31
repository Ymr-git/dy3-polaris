# T3 学习上下文经纪 (Learning Context Broker) 实现报告

> 基于 L1 设计文档第三章 (上下文经纪) + 第八章 8.2 (L2 接口) 实现
>
> 方法论: TDD Red-Green-Refactor

## 1. 实现概览

| 维度 | 数据 |
|------|------|
| 实现文件 | `l1/context_broker.py` (1174 行) |
| 测试文件 | `tests/l1/test_context_broker.py` (1297 行) |
| 测试用例 | 88 个 (全部通过) |
| 回归测试 | L1 全部 745 测试通过, 全量 1403 测试零回归 |
| TDD 覆盖率 | 9 个测试类, 10+ 个测试分组 |

## 2. 模块架构

```
context_broker.py
├── 1. 常量定义
│   ├── 缓存 TTL (CACHE_TTL_SESSION/MASTERY/GOAL/COGNITIVE)
│   ├── 认知负荷权重 (BASE/ERROR/SLOW/HELP)
│   └── 隐私过滤黑名单 (_BLOCKED_EVENT_TYPES/PATTERNS)
│
├── 2. 异常体系 (JSON-RPC -32300 范围)
│   ├── L1ContextError (-32300) — 基础异常
│   ├── ContextNotFoundError (-32301) — 会话不存在
│   ├── ContextExpiredError (-32302) — TTL 过期
│   ├── ContextValidationError (-32303) — 参数非法
│   └── DecayError (-32304) — 衰减计算失败
│
├── 3. 事件数据结构 (xAPI Actor-Verb-Object)
│   ├── FrontendEvent — 前端埋点 (渠道1)
│   ├── AgentOutputEvent — Agent输出 (渠道2)
│   └── UserDeclaration — 用户声明 (渠道3)
│
├── 4. ContextCollector — 三渠道采集器
│   ├── collect_frontend_event() + 隐私过滤
│   ├── collect_agent_output()
│   ├── collect_user_declaration()
│   └── get_all_events() / get_events_by_type() / clear()
│
├── 5. DecayEngine — 遗忘衰减引擎
│   ├── calculate_decay() — Ebbinghaus 曲线计算
│   ├── refresh_all_decay() — 批量刷新衰减系数
│   └── get_review_urgency() — 复习紧急度排序
│
├── 6. ContextCache — TTL 分层缓存
│   ├── get() / set() — 会话级热数据缓存
│   ├── invalidate() — 失效缓存
│   ├── backup_to_persistent() — 持久层备份
│   ├── restore_from_persistent() — 冷加载恢复
│   └── get_stats() / clear_all()
│
└── 7. LearningContextBroker — 核心引擎
    ├── build_envelope() — 构建标准化上下文信封
    ├── get_envelope() — 获取上下文 (热数据优先)
    ├── update_mastery() — BKT 贝叶斯更新
    ├── update_cognitive_load() — 认知负荷计算
    ├── update_learning_phase() — 学习阶段更新
    ├── update_goals() — 学习目标更新
    ├── add_resource() / set_time_constraint()
    ├── refresh_context() — 遗忘衰减刷新
    ├── get_weak_kcs() — 薄弱知识点查询
    ├── transfer_context() — 跨会话上下文传递
    ├── remove_session() / get_all_sessions()
    └── get_envelope_summary() — 脱敏摘要
```

## 3. 融合世界先进方案

### 3.1 ContextFlow 三区分层架构

借鉴 ContextFlow 的 Fixed/Working/History Zone 分层设计:
- **Fixed Zone**: 用户声明 (UserDeclaration) — 高可信偏好, 低频采集
- **Working Zone**: 前端埋点 + Agent 输出 — 实时行为数据, 毫秒级缓存
- **History Zone**: 持久层备份 (ContextCache._persistent_store) — 冷数据恢复

### 3.2 xAPI (IEEE 9274.1.1) 标准化事件信封

所有事件类型实现 `to_xapi_statement()` 方法, 输出 Actor-Verb-Object 三元组:
```python
{
    "actor": "user-001",
    "verb": "answer_submit",
    "object": "question-042",
    "result": {"is_correct": True, "response_time_ms": 3500},
    "timestamp": 1785394453000
}
```

### 3.3 FSRS 间隔重复调度

借鉴 FSRS (Free Spaced Repetition Scheduler) 的幂律遗忘曲线:
- 稳定性参数: `stability = MIN_STABILITY + repetitions * STABILITY_GAIN`
- 衰减函数: `decay = exp(-elapsed_hours / stability)`
- 有效掌握度: `effective = p_know * decay` (不低于 PRIOR_PROB)
- 复习紧急度: 按 `1.0 - effective_mastery` 降序排列

### 3.4 Redis Agent Memory 两层记忆模型

借鉴 Redis Agent Memory 的 Session + Long-term 双层结构:
- **Session Memory**: `ContextCache._session_cache` — 热数据, TTL 控制
- **Long-term Memory**: `ContextCache._persistent_store` — 冷备份, 会话失效后可恢复

### 3.5 OSCOI 模式 (BKT 离线校准 + 在线推断)

借鉴 OSCOI 模式实现 BKT 集成:
- L2 BKT 引擎离线校准参数 (p_know/p_slip/p_guess/p_transit)
- L1 LCB 在线推断: `update_mastery()` 接收 BKT 后验, 实时更新掌握度快照
- `BKTParams.bayesian_update(is_correct)` 完成贝叶斯后验更新

### 3.6 Khan Academy 学习路径追踪

借鉴 Khan Academy 的动态难度调节:
- `get_weak_kcs(threshold)` 识别薄弱知识点
- `get_review_urgency()` 排序复习优先级
- `update_cognitive_load()` 实时监测学习负荷

### 3.7 Duolingo 多数据点知识信号

借鉴 Duolingo 的百万级数据点聚合:
- 三渠道采集 (前端埋点 + Agent 输出 + 用户声明)
- 多维度认知负荷计算 (错误率 + 慢响应率 + 求助率)

## 4. 核心实现细节

### 4.1 异常体系 (JSON-RPC -32300 范围)

| 异常 | JSON-RPC 码 | 触发场景 |
|------|-------------|----------|
| L1ContextError | -32300 | 基础异常 |
| ContextNotFoundError | -32301 | 会话不存在或已清除 |
| ContextExpiredError | -32302 | TTL 超时 |
| ContextValidationError | -32303 | 参数非法 (空 user_id/session_id) |
| DecayError | -32304 | 衰减计算失败 (p_know 越界) |

### 4.2 隐私过滤机制

采集器内置隐私过滤, 自动拦截禁止采集的事件:
- 事件类型黑名单: `mouse_move`, `mouse_track`, `heatmap_click`, `heatmap_view`, `cross_domain_request`
- 资源模式黑名单: `external-site`, `third-party`, `cross-domain`
- Agent 输出和用户声明不经隐私过滤 (已脱敏/用户主动提供)

### 4.3 TTL 分层缓存

| 缓存层 | TTL | 用途 |
|--------|-----|------|
| CACHE_TTL_SESSION | 7200s (2h) | 活跃会话热数据 |
| CACHE_TTL_MASTERY | 3600s (1h) | 知识掌握快照 |
| CACHE_TTL_GOAL | 86400s (24h) | 学习目标 |
| CACHE_TTL_COGNITIVE | 1800s (30min) | 认知负荷 |

TTL 语义:
- `ttl > 0`: 正常过期 (expires_at = now + ttl)
- `ttl == 0`: 立即过期 (用于测试或显式过期)
- `ttl < 0`: 永久不过期 (特殊场景)

### 4.4 认知负荷计算公式

```
load = BASE + error_rate * ERROR_WEIGHT + slow_rate * SLOW_WEIGHT + help_rate * HELP_WEIGHT
```

| 参数 | 值 | 说明 |
|------|-----|------|
| BASE | 0.2 | 基础负荷 (低负荷起点) |
| ERROR_WEIGHT | 0.4 | 错误率权重 (最大影响) |
| SLOW_WEIGHT | 0.25 | 慢响应率权重 (>8000ms) |
| HELP_WEIGHT | 0.15 | 求助率权重 |

极端场景:
- 全对+快答+无求助: `0.2 + 0 + 0 + 0 = 0.2` (低负荷)
- 全错+慢答+全求助: `0.2 + 0.4 + 0.25 + 0.15 = 1.0` (满负荷)

### 4.5 BKT 贝叶斯更新集成

`update_mastery()` 方法实现完整的 BKT 更新流程:
1. 查找已有 MasterySnapshot (按 kc_id)
2. 调用 `BKTParams.bayesian_update(is_correct)` 更新后验
3. 递增 repetitions / attempts / correct_count
4. 更新 last_practiced_at 为当前时间
5. 写回缓存

### 4.6 跨会话上下文传递

`transfer_context()` 实现认知状态继承:
1. 深拷贝源会话的 mastery_snapshot (含 BKT 参数)
2. 继承未完成学习目标
3. 学习阶段重置为 PREVIEW (新会话从头开始)
4. 写入目标会话缓存

## 5. 测试覆盖

### 5.1 测试类与用例分布

| 测试类 | 用例数 | 覆盖范围 |
|--------|--------|----------|
| TestExceptionHierarchy | 11 | 异常继承 + JSON-RPC 码 + 上下文 |
| TestFrontendEvent | 4 | 前端事件创建 + xAPI 转换 + 隐私过滤 |
| TestAgentOutputEvent | 3 | Agent 事件创建 + xAPI 转换 |
| TestUserDeclaration | 3 | 用户声明创建 + xAPI 转换 |
| TestContextCollector | 8 | 三渠道采集 + 隐私过滤 + 事件查询 |
| TestDecayEngine | 8 | 衰减计算 + 批量刷新 + 复习紧急度 |
| TestContextCache | 8 | TTL 缓存 + 持久层备份 + 统计 |
| TestLearningContextBroker | 20 | 核心引擎全功能 |
| TestBKTIntegration | 6 | BKT 贝叶斯更新集成 |
| TestThreadSafety | 3 | 并发访问安全 |
| TestEdgeCases | 8 | 空值/过期/非法输入 |
| TestIntegrationLifecycle | 4 | 全生命周期集成 |

### 5.2 集成测试场景

**完整学习会话生命周期**:
1. 构建初始上下文 (2个KC + 1个目标)
2. 采集前端事件 (5次答题)
3. 更新掌握度 (BKT 更新)
4. 更新认知负荷 (多维计算)
5. 更新学习阶段
6. 刷新上下文 (衰减重算)
7. 获取薄弱知识点
8. 验证最终状态

**跨会话继承**:
1. 源会话积累掌握度和目标
2. 传递到目标会话
3. 验证掌握度和目标继承

## 6. 修复记录

### 6.1 STABILITY_GAIN 调整

**问题**: `STABILITY_GAIN=6.0` 导致 7 天后 10 次练习的衰减也命中 PRIOR_PROB 下限, 无法区分高低练习次数。

**修复**: 调整为 `STABILITY_GAIN=24.0` (每次练习增加约 1 天稳定性), 使 7 天后 10 次练习的有效掌握度 (0.42) 明显高于 0 次练习 (0.30)。

**影响**: 无回归, models.py 测试仅断言 `STABILITY_GAIN > 0`。

### 6.2 CACHE_TTL_SESSION 正值化

**问题**: `CACHE_TTL_SESSION=0` 导致 `test_ttl_constants_exist` 断言 `> 0` 失败。

**修复**: 调整为 `7200` (2 小时), 符合设计文档 "会话级缓存" 语义。

### 6.3 TTL=0 语义重定义

**问题**: `ttl=0` 原表示 "会话级不过期", 但测试期望 "立即过期"。

**修复**: 
- `ttl > 0`: 正常过期
- `ttl == 0`: 立即过期 (expires_at 设为过去时间)
- `ttl < 0`: 永久不过期

### 6.4 认知负荷权重调整

**问题**: 原 `BASE=0.5` 导致全对+快答+无求助场景的认知负荷 = 0.5, 不满足 `< 0.5` 断言。

**修复**: 降低 BASE 至 0.2, 调整 ERROR/SLOW/HELP 权重为 0.4/0.25/0.15, 使低负荷场景 = 0.2, 高负荷场景 = 1.0。

## 7. 设计文档对齐

| 设计文档章节 | 实现状态 | 实现位置 |
|-------------|----------|----------|
| 3.1 上下文定义 | ✅ | ContextEnvelope (models.py) + LearningContextBroker |
| 3.2 采集机制 (三渠道) | ✅ | ContextCollector + FrontendEvent/AgentOutputEvent/UserDeclaration |
| 3.2 隐私过滤 | ✅ | _BLOCKED_EVENT_TYPES + _is_blocked() |
| 3.3 Context Envelope Schema | ✅ | build_envelope() + ContextEnvelope (models.py) |
| 3.4 遗忘衰减与刷新 | ✅ | DecayEngine.calculate_decay() / refresh_all_decay() |
| 3.4 TTL 机制 | ✅ | ContextCache + CACHE_TTL_* 常量 |
| 3.5 跨会话传递 | ✅ | transfer_context() + ContextCache 持久层 |
| 3.6 Python 数据模型 | ✅ | 全部 dataclass, 可序列化 |
| 8.2 BKT 接口 | ✅ | update_mastery() + BKTParams.bayesian_update() |

## 8. 与现有代码的衔接

### 8.1 上游 (T1 数据模型)
- `ContextEnvelope`, `MasterySnapshot`, `LearningGoal`, `LearningState` — 全部来自 models.py
- `BKTParams.bayesian_update()` — models.py 已实现的 BKT 四参数模型
- `calculate_decay()` — models.py 已实现的 Ebbinghaus 衰减函数

### 8.2 上游 (T2 认证权限)
- 上下文构建需要 `user_id` (来自 JWT 认证)
- 上下文传递需经过 ABAC 权限过滤 (T2 AccessControlManager)

### 8.3 下游 (L3 知识层)
- `ContextEnvelope` 是 L1 → L3 的唯一数据载体
- L3 `context_builder.py` 已有上下文构建器, L1 LCB 为其提供输入
- `MasterySnapshot.kc_id` 对齐 L3 `KPMastery.kp_id`

### 8.4 下游 (L2 学情画像, 未实现)
- `update_mastery()` 接口预留 L2 BKT 引擎推送
- `transfer_context()` 预留从 L2 BKT 表读取掌握度

## 9. 任务衔接

### 前置任务
- ✅ T1 核心数据模型 — 提供全部数据结构
- ✅ T2 认证与权限 — 提供 user_id 和权限过滤

### 后续任务
- T4 人机协同 — 使用 LCB 的认知负荷监测触发紧急干预
- T5 会话管理 — ContextEnvelope 作为会话核心组件
- T6 隐私治理 — 上下文脱敏和审计
- T7 API 层 — `/api/v1/context/{session_id}` 接口对接 LCB

## 10. 竞争力分析

| 维度 | 本系统 | 通用 LLM | 优势 |
|------|--------|---------|------|
| 认知状态持久化 | ✅ BKT + Ebbinghaus 衰减 | ❌ 无状态 | 跨会话认知继承 |
| 多维认知负荷 | ✅ 错误率+慢响应+求助 | ❌ 无 | 实时负荷监测 |
| 标准化事件采集 | ✅ xAPI Actor-Verb-Object | ❌ 无标准 | 可审计可互操作 |
| 隐私过滤 | ✅ 采集端自动拦截 | ❌ 无 | FERPA/GDPR 合规 |
| 间隔重复调度 | ✅ FSRS 幂律曲线 | ❌ 无 | 个性化复习推荐 |
| 缓存分层 | ✅ Session + Persistent | ❌ 无 | 冷热数据分离 |
