# L1 用户域实现任务拆分

> 基于 `02-设计/L1-用户域设计/layer1-user-domain.html`（8 章设计文档），将用户域拆分为 7 个可衔接的实现任务。
>
> 当前状态：L1 层仅有设计文档，无代码实现。L0/L3/L4/L5/L6 层已有代码。

## 依赖关系

```
T1 (核心数据模型)
  └─ T2 (认证与权限)
       └─ T3 (上下文经纪)
            ├─ T4 (人机协同)     ─┐
            ├─ T5 (会话管理)     ─┤ T7 (API层与集成)
            └─ T6 (隐私治理)     ─┘
```

---

## T1: 核心数据模型 (Phase 1 — 基础)

### 设计依据
- 第二章 2.1-2.3: 角色枚举、ABAC 属性维度
- 第三章 3.1, 3.3, 3.6: 上下文定义、Context Envelope Schema、Python 数据模型
- 第五章 5.1, 5.3: 会话模型、Fork 数据结构
- 第七章 7.3: ER 图（User / Role / Session / LearningContext / AuditLog 五张表）

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/models.py` | 全部枚举 + dataclass/pydantic 模型 |

### 核心模型清单
1. **角色与权限**
   - `UserRole` 枚举: UNDERGRAD / GRADUATE / TEACHER / ADMIN / ALUMNI
   - `UserStatus` 枚举: ACTIVE / SUSPENDED / ALUMNI
   - `ABACAttributes` dataclass: grade_level, major_direction, course_progress, lab_access_tier, supervisor_id
   - `User` dataclass: user_id, student_id, institution_id, role, status, abac_attributes
   - `Permission` 枚举: 12 项功能权限（知识库读/写、Agent 调用、学情查看等）

2. **学习上下文**
   - `LearningPhase` 枚举: PREVIEW / PRACTICE / QUIZ / REVIEW
   - `MasterySnapshot` dataclass: kc_id, p_know, last_practiced_at, decay_factor, repetitions
   - `LearningGoal` dataclass: description, priority, deadline
   - `LearningState` dataclass: phase, session_duration_ms, interaction_count, cognitive_load
   - `ContextEnvelope` dataclass: envelope_id, user_id, session_id, timestamp, learning_state, mastery_snapshot, goals, ttl
     - 方法: `is_expired()`, `refresh_decay()`, `get_weak_kcs()`

3. **会话与 Fork**
   - `SessionType` 枚举: DIAGNOSIS / LEARNING / LAB_GUIDE / ASSESSMENT
   - `SessionStatus` 枚举: ACTIVE / PAUSED / FORKED / COMPLETED
   - `LearningSession` dataclass: session_id, user_id, session_type, parent_session_id, fork_point_seq, context, agent_states, interaction_log, artifacts, status, checkpoint_indices
   - `SessionFork` dataclass: fork_id, source_session_id, fork_point_seq, fork_reason, branch_label, snapshot_at_fork, merge_target, is_merged

4. **审计与脱敏**
   - `DataLevel` 枚举: L1(公开) / L2(内部) / L3(敏感) / L4(机密)
   - `AuditAction` 枚举: VIEW / EXPORT / MODIFY / AGENT_INVOKE / LOGIN / LOGOUT
   - `AuditLogEntry` dataclass: log_id, actor_id, actor_role, action, target_resource, target_data_level, purpose, result, session_id, created_at

5. **HiTL 相关**
   - `HiTLType` 枚举: CONFIRMATION / CORRECTION / CREATIVE / EMERGENCY
   - `HiTLPriority` 枚举: P0 / P1 / P2 / P3
   - `ConfidenceGate` 枚举: PASS(≥0.85) / WARNING(0.4~0.85) / BLOCK(<0.4)
   - `ApprovalRequest` / `ApprovalResponse` / `FeedbackReport` / `EmergencyAlert` dataclass

### 依赖
- 无（地基任务）
- 被下游所有任务依赖

### 与现有代码的衔接
- `ContextEnvelope.mastery_snapshot` 对齐 L3 `api_models.py` 中的 `KPMastery` 结构（kc_id ↔ kp_id, p_know ↔ mastery_prob）
- `LearningGoal` 对齐 L3 `BloomLevel` 枚举

---

## T2: 认证与权限控制 (Phase 2 — 访问控制)

### 设计依据
- 第二章 2.2: 角色权限矩阵（12 项功能 × 4 角色）
- 第二章 2.3: ABAC 属性策略 + Cedar 策略语言示例
- 第二章 2.4: 角色生命周期（注册→审核→激活→变更→毕业→归档）
- 第七章 7.2: API `/api/v1/auth/*`, `/api/v1/users/*`

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/auth.py` | JWT 签发/验证/撤销 + 用户注册/激活生命周期 |
| `l1/access_control.py` | RBAC 权限矩阵 + ABAC 策略评估引擎 |

### 核心实现

#### 2.1 JWT 认证 (`auth.py`)
- `JWTManager` 类:
  - `issue_token(user) -> str`: 签发 JWT，payload 含 user_id, role, abac_attributes
  - `verify_token(token) -> User`: 验证签名 + 过期 + Redis 黑名单
  - `revoke_token(token)`: 加入 Redis 黑名单（即时生效）
  - `refresh_token(token) -> str`: 刷新 Token
- 密码安全: bcrypt 哈希存储
- Token 过期: 默认 2 小时，可配置

#### 2.2 RBAC 权限矩阵 (`access_control.py`)
- `RBACMatrix` 类:
  - 内置 12 项功能权限 × 4 角色的权限表
  - `check_permission(role, permission) -> bool`
  - 支持条件标记（如"限导师授权范围"、"仅课程范围"）

#### 2.3 ABAC 策略评估 (`access_control.py`)
- `ABACEvaluator` 类:
  - `evaluate(user, action, resource, context) -> bool`
  - 基于 Python 实现的轻量策略引擎（Cedar 语义子集）
  - 策略示例:
    - 研究生仅可访问导师授权范围的实验数据
    - 本科生知识生成 Agent 调用频率限制
    - 课程进度 < 0.3 时不推荐综合实验指导
  - 支持动态属性查询（course_progress 从 L2 获取）

#### 2.4 角色生命周期
- `UserLifecycleManager` 类:
  - `register(student_id, ...) -> User`: 创建用户记录
  - `activate(user_id)`: 审核通过后激活，签发 Token
  - `change_role(user_id, new_role)`: 权限变更（如本科→研究生）
  - `graduate(user_id)`: 降级为 ALUMNI（只读）
  - `archive(user_id)`: 数据归档/删除

### 依赖
- T1 (数据模型)

### 与现有代码的衔接
- 对接 L0 `governance/audit_engine.py`: 认证操作写入审计日志
- 对接 L0 `governance/compliance.py`: 合规策略拉取
- `access_control.py` 与 L3 `access_control.py` 模式对齐（L3 已有同名模块，L1 为用户域级别的权限控制）

---

## T3: 学习上下文经纪 (Phase 3 — 核心创新)

### 设计依据
- 第三章 3.1-3.6: 上下文定义、采集机制、传递协议、衰减与刷新、Python 数据模型
- 第八章 8.2: 与 L2 的接口（学情画像输入/输出、BKT 参数更新）

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/context_broker.py` | LCB 核心引擎：采集、构建、衰减、刷新、传递 |

### 核心实现

#### 3.1 上下文采集 (`ContextCollector`)
- 三渠道采集:
  - `collect_frontend_event(event)`: 前端埋点（页面浏览、答题响应、资源完成度）
  - `collect_agent_output(agent_result)`: Agent 输出推断（BKT 更新、路径推荐）
  - `collect_user_declaration(declaration)`: 用户显式声明（可用时间、偏好、困惑点）
- 采集约束: 不采集鼠标轨迹、热力图、跨域数据

#### 3.2 上下文构建与管理 (`LearningContextBroker`)
- `build_envelope(user_id, session_id) -> ContextEnvelope`: 构建标准化上下文信封
- `get_envelope(session_id) -> ContextEnvelope`: 获取当前上下文（Redis 热数据优先）
- `update_mastery(session_id, kc_id, p_know)`: 更新知识掌握快照
- `update_cognitive_load(session_id, interactions)`: 计算认知负荷（响应时间 + 错误率 + 求助频率）
- `get_weak_kcs(session_id, threshold) -> list[str]`: 获取薄弱知识点

#### 3.3 遗忘衰减引擎 (`DecayEngine`)
- `calculate_decay(p_know, last_practiced, repetitions, current_ts) -> float`:
  - 基于 Ebbinghaus 遗忘曲线
  - 稳定性参数: `stability = MIN_STABILITY + repetitions * STABILITY_GAIN`
  - 衰减: `decay = exp(-elapsed_hours / stability)`
  - 有效掌握度: `effective = p_know * decay`（不低于先验概率）
- `refresh_all_decay(envelope) -> ContextEnvelope`: 批量刷新衰减系数

#### 3.4 TTL 与缓存分层
- TTL 配置:
  - 学习状态: 会话级（随 Session 关闭失效）
  - 知识掌握快照: 1 小时
  - 学习目标: 24 小时
  - 认知负荷: 30 分钟
  - 遗忘衰减: 持续计算
- `ContextCache` 类:
  - Redis 层: 热数据缓存（毫秒级读取）
  - PostgreSQL 层: 持久化快照（冷加载恢复）
  - `get(session_id) -> ContextEnvelope | None`
  - `set(session_id, envelope, ttl)`
  - `invalidate(session_id)`

#### 3.5 跨会话上下文传递
- `transfer_context(source_session, target_session)`: 会话间上下文继承
  - 知识掌握度: 从 L2 BKT 表读取
  - 学习目标: 未完成目标自动继承
  - 交互摘要: 前一会话的聚合摘要

### 依赖
- T1 (数据模型)
- T2 (权限过滤：上下文需经过权限过滤后才能传递给下层)

### 与现有代码的衔接
- 对接 L2（未实现）: BKT 参数更新接口
- `ContextEnvelope` 是 L1 → L2/L3/L4 的唯一数据载体
- 对接 L3 `context_builder.py`: L3 已有上下文构建器，L1 LCB 为其提供输入

---

## T4: 人机协同设计 (Phase 4 — 并行核心服务)

### 设计依据
- 第四章 4.1-4.7: 协同场景分类、触发条件、交互模式、置信度门控、反馈回路
- 第八章 8.4: 与 CC2 Plan-Approval Gate 的接口

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/hitl_manager.py` | HiTL 协同管理器：四类场景 + 置信度门控 + 反馈回路 |

### 核心实现

#### 4.1 协同场景管理 (`HiTLManager`)
- 四类协同:
  - `handle_confirmation(content, user_id)`: 确认型 — 学生确认"已理解"
  - `handle_correction(feedback, user_id)`: 纠错型 — 学生标记"不理解" → Agent 自纠 → 3 次未解决 → 教师介入
  - `handle_creative(request, teacher_id)`: 创造型 — 教师创建内容 → 审核 Agent 校验 → 可选同行评议
  - `handle_emergency(session_id, trigger)`: 紧急干预 — 自动暂停 + 通知教师

#### 4.2 置信度门控 (`ConfidenceGate`)
- `evaluate(confidence) -> GateResult`:
  - `confidence >= 0.85` → PASS: 直接呈现，后台记录 Provenance
  - `0.4 <= confidence < 0.85` → WARNING: 附置信度标签，建议人工复核
  - `confidence < 0.4` → BLOCK: 阻止呈现，触发审核流程
- 与 L5 `reflection_quality.py` 的 `QualityGate` 对接（已有实现）

#### 4.3 紧急干预检测 (`EmergencyDetector`)
- 触发条件:
  - 连续错误 >= 10 次
  - 认知负荷 >= 0.95
  - 异常答题速度 (< 5 秒/题)
- `check(session_id, context) -> EmergencyAlert | None`
- 自动暂停会话 + 通知教师 + 记录审计

#### 4.4 反馈回路 (`FeedbackLoop`)
- `submit_feedback(feedback_report)`: 接收学生/教师反馈
- 反馈分类:
  - 事实性错误 → 触发知识库修正申请（L3）
  - 适配性不足 → 触发 ABAC 策略调整建议
  - 安全问题 → 升级至 L0 治理
- `get_feedback_history(session_id) -> list[FeedbackReport]`

#### 4.5 交互模式
- `PassiveConfirmation`: 内容底部附加确认控件
- `ProactiveSuggestion`: 检测学习瓶颈时弹出建议
- `MandatoryBlock`: 强制暂停 + 休息提示
- `OptionalNegotiation`: 教师与系统双向协商

### 依赖
- T1 (HiTL 数据模型)
- T2 (权限：确定审批方)
- T3 (认知负荷监测来自 LCB)

### 与现有代码的衔接
- 对接 L0 `cc2/engine.py`: CC2 Plan-Approval Gate 改造为 HiTL Gate
- 对接 L5 `reflection_quality.py`: 置信度由 CC1Reviewer 输出
- 对接 L4 `feedback_aggregator.py`: 反馈数据聚合

---

## T5: 学习会话管理 (Phase 4 — 并行核心服务)

### 设计依据
- 第五章 5.1-5.5: 会话模型、会话类型、Session Fork、会话间上下文传递
- 第七章 7.2: API `/api/v1/sessions/*`

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/session_manager.py` | 学习会话生命周期管理 + Fork 机制 |

### 核心实现

#### 5.1 会话生命周期 (`LearningSessionManager`)
- `create_session(user_id, session_type) -> LearningSession`: 创建会话，加载上下文
- `get_session(session_id) -> LearningSession`: 获取会话详情
- `pause_session(session_id)`: 暂停会话，持久化状态
- `resume_session(session_id)`: 恢复会话
- `complete_session(session_id)`: 完成会话，更新 BKT 参数，归档交互日志
- `add_interaction(session_id, interaction)`: 记录交互（问答/答题/反馈）
- `add_artifact(session_id, artifact)`: 关联产出物

#### 5.2 Checkpoint 管理 (`CheckpointManager`)
- `create_checkpoint(session) -> int`: Agent 完成完整推理后自动创建检查点
  - 快照内容: 上下文 + Agent 状态 + 交互历史 + 产出物
- `list_checkpoints(session_id) -> list[Checkpoint]`
- `load_checkpoint(session_id, seq) -> LearningSession`: 加载检查点状态

#### 5.3 Session Fork (`ForkManager`)
- `fork(session_id, fork_point_seq, reason, label) -> LearningSession`:
  - 从指定 Checkpoint 创建分支会话
  - 继承 Fork 点的完整状态
  - 独立的上下文、交互历史和产出物
- `merge(fork_session_id, target_session_id)`:
  - 合并 Fork 分支的掌握度更新回主会话
  - 丢弃另一分支
- `list_forks(session_id) -> list[SessionFork]`
- `discard_fork(fork_session_id)`: 丢弃分支，状态归档

#### 5.4 会话间上下文传递
- `inherit_context(source_session, target_session)`:
  - 知识掌握度: 从 L2 BKT 表读取
  - 学习目标: 未完成目标自动继承
  - 交互摘要: 聚合后传入
- 后台遗忘衰减任务: 即使无活跃会话也持续运行

#### 5.5 会话状态持久化
- Redis 层: 会话热数据（ACTIVE 状态，毫秒级读取）
- PostgreSQL 层: 会话冷数据（PAUSED/COMPLETED，持久化存储）

### 依赖
- T1 (会话数据模型)
- T2 (用户身份验证)
- T3 (Context Envelope 作为会话核心组件)

### 与现有代码的衔接
- 对接 L5 `session_manager.py`: 已有 SessionManager 实现，L1 需在其基础上扩展学习场景功能
- L5 已有 `trigger_fork_merge_review()` 方法，L1 Fork 合并后可调用
- 对接 L5 `kernel_persistence.py`: 持久化机制对齐

---

## T6: 隐私保护与数据治理 (Phase 4 — 并行核心服务)

### 设计依据
- 第六章 6.1-6.4: 数据分类分级、数据最小化、脱敏方法、留存策略
- 第七章 7.2: API `/api/v1/audit/logs`, `/api/v1/export/learner-data`
- 第八章 8.1: 与 L0 的接口（审计日志上报、隐私事件通知）

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/privacy_governance.py` | 数据分级 + 脱敏 + 留存 + 审计日志 + 隐私事件 |

### 核心实现

#### 6.1 数据分级控制 (`DataClassifier`)
- 四级分类: L1(公开) / L2(内部) / L3(敏感) / L4(机密)
- `classify(data_type) -> DataLevel`
- `check_access(user, data_level) -> bool`: 根据用户角色和 ABAC 属性检查访问权限
- 数据最小化校验: 采集前校验是否在"必须采集"清单内

#### 6.2 数据脱敏引擎 (`DesensitizationEngine`)
- `desensitize(data, data_level, method) -> Any`:
  - 学号: SHA-256 哈希 + 盐值 → `a3f2...b7c1`（前4后4）
  - 答题记录: 聚合为正确率
  - 响应时间: 分桶泛化（"正常"/"偏快"/"偏慢"）
  - 学习路径: 差分隐私加噪
  - 学情报告: 移除姓名/学号，替换为伪 ID
- `anonymize_for_research(user_data, k=5, l=3)`: K-匿名 + l-多样性

#### 6.3 数据留存策略 (`RetentionManager`)
- 四阶段留存:
  - 阶段一（课程期间）: 原始数据，完整保留
  - 阶段二（毕业后 1 年）: 学号脱敏，答题记录聚合
  - 阶段三（毕业后 1 年起）: 完全匿名化（K-匿名 ≥ 5, l-多样性 ≥ 3）
  - 永久删除（毕业后 3 年 / 用户主动申请）
- `check_retention(user_id) -> RetentionAction`: 检查用户数据的留存阶段
- `execute_retention(user_id, action)`: 执行留存操作

#### 6.4 审计日志管理 (`AuditLogger`)
- `log(entry: AuditLogEntry)`: 写入审计日志（append-only）
- `query(filters) -> list[AuditLogEntry]`: 分页查询审计日志
  - 支持按 actor_id / action / data_level / time_range 筛选
  - 教师仅能查询其课程范围内的日志
- 异步批量写入（每 100 条或每 30 秒），不阻塞主流程
- 定期归档至冷存储

#### 6.5 隐私事件通知 (`PrivacyEventNotifier`)
- `notify(event_type, user_id, data_level, detail)`:
  - 检测到数据异常访问 → 通知 L0
  - 用户申请数据删除 → 通知 L0
  - 留存策略执行 → 通知 L0

### 依赖
- T1 (数据模型: DataLevel, AuditLogEntry)
- T2 (权限: 访问控制)

### 与现有代码的衔接
- 对接 L0 `governance/audit_engine.py`: 审计日志上报至 L0
- 对接 L0 `governance/compliance.py`: 合规策略执行
- 对接 L0 `governance/policy_store.py`: 隐私策略拉取
- 审计日志格式对齐 L0 `governance/models.py` 中的审计模型

---

## T7: API 层与层间集成 (Phase 5 — 聚合出口)

### 设计依据
- 第七章 7.1-7.5: 技术选型、API 设计、ER 图、关键流程伪代码、性能指标
- 第八章 8.1-8.4: 与 L0/L2/L3/CC2 的接口定义

### 交付物
| 文件 | 内容 |
|------|------|
| `l1/api/router.py` | FastAPI 路由定义（17 个核心接口） |
| `l1/api/middleware.py` | JWT 认证中间件 + ABAC 权限校验中间件 + 审计日志中间件 |
| `l1/interfaces.py` | 层间接口实现（L0/L2/L3/CC2 四组接口） |
| `l1/__init__.py` | L1 模块入口 + 依赖注入容器 |

### 核心实现

#### 7.1 RESTful API 路由 (`router.py`)
17 个核心接口:

| 路径 | 方法 | 功能 | 对应任务 |
|------|------|------|----------|
| `/api/v1/auth/login` | POST | 学号+密码登录 | T2 |
| `/api/v1/auth/logout` | POST | 登出，Token 加入黑名单 | T2 |
| `/api/v1/auth/refresh` | POST | 刷新 JWT Token | T2 |
| `/api/v1/users/me` | GET | 获取当前用户信息与 ABAC 属性 | T2 |
| `/api/v1/users/me/preferences` | PUT | 更新学习偏好 | T2 |
| `/api/v1/sessions` | POST | 创建学习会话 | T5 |
| `/api/v1/sessions/{id}` | GET | 获取会话详情 | T5 |
| `/api/v1/sessions/{id}/fork` | POST | 创建 Session Fork | T5 |
| `/api/v1/sessions/{id}/merge` | POST | 合并 Fork 分支 | T5 |
| `/api/v1/sessions/{id}/pause` | POST | 暂停会话 | T5 |
| `/api/v1/context/{session_id}` | GET | 获取 Context Envelope | T3 |
| `/api/v1/context/{session_id}/refresh` | POST | 刷新上下文 | T3 |
| `/api/v1/hitl/confirm` | POST | HiTL 确认型操作 | T4 |
| `/api/v1/hitl/feedback` | POST | HiTL 纠错型反馈 | T4 |
| `/api/v1/hitl/emergency` | GET | 获取紧急干预状态 | T4 |
| `/api/v1/audit/logs` | GET | 查询审计日志 | T6 |
| `/api/v1/export/learner-data` | GET | 导出脱敏学情数据 | T6 |

#### 7.2 中间件 (`middleware.py`)
- `AuthMiddleware`: JWT 验证 + Redis 黑名单检查
- `ABACMiddleware`: ABAC 策略实时评估
- `AuditMiddleware`: 自动记录请求审计日志（异步）

#### 7.3 层间接口实现 (`interfaces.py`)
- **L1 → L0** (上报):
  - `report_audit_logs(entries)`: 审计日志批量上报
  - `report_privacy_event(event)`: 隐私事件通知
  - `write_provenance(record)`: Provenance 写入
- **L0 → L1** (拉取):
  - `pull_compliance_policies()`: 合规策略拉取
  - `receive_policy_update(update)`: 策略变更通知
- **L1 → L2** (传递):
  - `send_context_envelope(envelope)`: 上下文信封传递
  - `send_memory_entry(entry)`: 学习记忆写入
  - `send_decay_request(request)`: 遗忘调度请求
- **L2 → L1** (返回):
  - `receive_learner_profile(profile)`: 学情画像输出
  - `receive_bkt_update(update)`: BKT 参数更新
- **L1 → L3** (请求):
  - `check_access(check)`: 知识访问权限校验
  - `request_resources(request)`: 学习资源推荐请求
- **L3 → L1** (返回):
  - `receive_knowledge_result(result)`: 知识查询结果
- **L1 ↔ CC2** (HiTL Gate):
  - `route_approval_request(request)`: 确认请求路由
  - `route_approval_response(response)`: 确认响应路由
  - `report_feedback(feedback)`: 反馈数据上报
  - `alert_emergency(alert)`: 紧急干预通知

#### 7.4 核心交互流程
```python
async def learning_interaction_flow(user_id, request):
    # Step 1: 身份验证与权限校验
    user = await verify_jwt_token(request.token)
    if is_token_revoked(request.token): raise AuthenticationError
    if not abac_evaluate(user, "invoke_agent", request.agent_type):
        raise PermissionDenied
    write_audit_log(user, "AGENT_INVOKE", request.agent_type)

    # Step 2: 上下文加载/刷新
    session = await get_or_create_session(user_id)
    context = await load_context(session.session_id)
    if context.is_expired():
        context = await refresh_context(session.session_id)
        context.refresh_decay(current_ts_ms())

    # Step 3: 认知负荷检查 → 紧急干预
    if context.learning_state.cognitive_load >= EMERGENCY_THRESHOLD:
        await trigger_emergency_pause(session, "认知负荷过高")
        await notify_teacher(user_id, "学生认知负荷超标")
        return EmergencyResponse(reason="建议休息后继续")

    # Step 4: 构建 Agent 请求
    agent_request = build_agent_request(
        agent_type=request.agent_type,
        context=context,
        user_input=request.content,
        weak_kcs=context.get_weak_kcs()
    )

    # Step 5: Agent 调用 (L4/L5)
    agent_result = await invoke_agent_pipeline(agent_request)

    # Step 6: 置信度门控
    if agent_result.confidence < BLOCK_THRESHOLD:
        return BlockedResponse(escalation="TEACHER_REVIEW")
    elif agent_result.confidence < WARNING_THRESHOLD:
        agent_result.confidence_label = "WARNING"
        agent_result.requires_confirmation = True

    # Step 7: 返回结果 + 收集反馈
    return LearningResponse(
        content=agent_result.content,
        confidence=agent_result.confidence,
        confirmation_required=agent_result.requires_confirmation,
        context_summary=context.to_summary(),
        feedback_options=["UNDERSTOOD", "NEED_MORE", "INCORRECT", "REPORT"]
    )
```

#### 7.5 性能指标
| 指标 | 目标值 |
|------|--------|
| 并发在线用户 | ≥ 500 |
| API P50 延迟 | ≤ 200ms |
| API P99 延迟 | ≤ 800ms |
| 上下文热加载 | ≤ 50ms |
| 上下文冷加载 | ≤ 300ms |
| JWT 验证延迟 | ≤ 5ms |
| Session Fork 创建 | ≤ 1s |
| 审计日志写入 | ≤ 10ms |

### 依赖
- T1-T6 全部任务

### 与现有代码的衔接
- 对接 L6 `api/router.py`: L6 已有 API 路由，L1 API 作为补充
- 对接 L3 `api/router.py`: L3 已有 API 路由
- 对接 L5 全部模块: Agent 调用通过 L5 编排引擎
- 对接 L4 `decision_engine.py`: 学习决策通过 L4

---

## 实施顺序与预估

| 阶段 | 任务 | 预估工作量 | 可并行 |
|------|------|-----------|--------|
| Phase 1 | T1 核心数据模型 | 1-2 天 | 否（地基） |
| Phase 2 | T2 认证与权限 | 2-3 天 | 否 |
| Phase 3 | T3 上下文经纪 | 3-4 天 | 否 |
| Phase 4a | T4 人机协同 | 2-3 天 | 是（与 T5/T6 并行） |
| Phase 4b | T5 会话管理 | 2-3 天 | 是（与 T4/T6 并行） |
| Phase 4c | T6 隐私治理 | 2-3 天 | 是（与 T4/T5 并行） |
| Phase 5 | T7 API与集成 | 3-4 天 | 否（聚合） |
| **总计** | | **15-22 天** | |

## 测试策略

每个任务配套测试文件，位于 `tests/l1/`:
- `test_models.py` (T1): 枚举完备性、dataclass 序列化、ContextEnvelope 方法
- `test_auth.py` (T2): JWT 签发/验证/撤销、RBAC 矩阵、ABAC 策略评估
- `test_context_broker.py` (T3): 上下文采集、衰减计算、TTL 过期、缓存分层
- `test_hitl.py` (T4): 四类协同场景、置信度门控、紧急干预、反馈回路
- `test_session.py` (T5): 会话生命周期、Checkpoint、Fork 创建/合并
- `test_privacy.py` (T6): 数据分级、脱敏方法、留存策略、审计日志查询
- `test_api.py` (T7): API 端到端测试、中间件链、层间接口
