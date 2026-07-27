# T1: 意图理解与上下文构建 -- 设计书

## 1. 概述

T1 是 L3 领域知识层面向 L4 决策引擎的核心入口模块，负责将用户查询转化为结构化的检索上下文，并驱动意图路由检索。

**两个核心组件**：
- `ContextBuilder` -- 上下文构建器（四阶段流水线）
- `IntentRouter v2` -- 上下文增强路由引擎

**文件位置**：
- `src/dy3_polaris/l3/context_builder.py` (~980 行)
- `src/dy3_polaris/l3/intent_router.py` (~1180 行)
- `tests/test_l3_context_builder.py` (127 测试)
- `tests/test_l3_intent_router_v2.py` (85 测试)

## 2. 设计参考

| 方案 | 借鉴要点 | 应用位置 |
|------|----------|----------|
| ReCAP (NeurIPS 2025) | 递归上下文重注入、滑动窗口记忆 | HistoryCompressor、IntentRouter._route_single |
| Self-RAG (ICLR 2024) | [Retrieve] Token 检索需求评估 | RetrievalNeedAssessor、route_with_context 跳过逻辑 |
| Agentic RAG | OODA 循环、动态路由选择 | IntentRouter 四路意图路由 |
| Context Recycling (2026) | 五层记忆架构、Token 预算管理 | ContextBudget、HistoryCompressor |
| Plan-and-Solve | 计划作为骨架、多步查询 | _route_multi_query 多变体 RRF 融合 |
| SEAL (Neurocomputing 2025) | Agent 校准、KG 结构对齐 | SchemaContextInjector、_apply_schema_boost |
| Kaman 3.0 (2026) | 自适应上下文管理、KMMS 五层认知 | ContextBudget、LearnerContextAdapter |

## 3. ContextBuilder 架构

### 3.1 四阶段构建流程

```
输入 → Phase 1: 输入预处理 → Phase 2: 意图理解 → Phase 3: 上下文组装 → Phase 4: 自我评估 → QueryContext
```

**Phase 1 -- 输入预处理**
- CoreferenceResolver：指代消解（"它" → "Dy3+"）
- 查询清洗：去除多余空格、标点标准化

**Phase 2 -- 意图理解**
- EntityExtractor：领域 NER（离子/化学式/光谱项/数值/关键词）
- SchemaContextInjector：检测领域 → 注入 KG Schema 上下文
- LLMClassifier（协议，预留接口）：LLM 意图提示

**Phase 3 -- 上下文组装**
- HistoryCompressor：三种压缩策略（RECENT/SUMMARIZE/SLIDING_WINDOW）
- LearnerContextAdapter：学习者画像适配（bloom 层级 → 深度/top_k）
- QueryRewriter：查询重写（5 种策略：同义词/分解/HyDE/扩展/压缩）

**Phase 4 -- 自我评估**
- RetrievalNeedAssessor：是否需要检索（Self-RAG 风格）
- 参数建议：suggested_top_k / suggested_depth
- 意图提示推断：基于实体和领域

### 3.2 核心数据结构 -- QueryContext

```python
@dataclass
class QueryContext:
    context_id: str           # 唯一标识 (ctx-xxxx)
    original_query: str       # 原始查询
    resolved_query: str       # 指代消解后
    rewritten_queries: list   # 查询重写变体
    intent_hint: str          # 意图提示 (如 "numeric+relational")
    entities: list[str]       # 提取的实体
    dialog_history: list      # 压缩后的对话历史
    learner_adaptation: dict  # 学习者适配
    schema_context: str       # KG Schema 上下文
    domain: str               # 检测到的领域
    needs_retrieval: bool     # 是否需要检索
    suggested_top_k: int      # 建议返回数
    suggested_depth: int      # 建议图深度
    metadata: dict            # 构建元信息

    @property
    def active_query(self) -> str:  # 返回 resolved_query
```

### 3.3 子组件

| 组件 | 职责 | 关键方法 |
|------|------|----------|
| ContextBudget | Token 预算分配 | `budget_for(key)` |
| HistoryCompressor | 对话历史压缩 | `compress(turns)` |
| CoreferenceResolver | 指代消解 | `resolve(query, turns)` |
| SchemaContextInjector | KG Schema 注入 | `inject(query, domain)` |
| LearnerContextAdapter | 学习者适配 | `adapt(profile)` |
| RetrievalNeedAssessor | 检索需求评估 | `assess(query, ctx)` |

## 4. IntentRouter v2 架构

### 4.1 意图分类

**IntentType 枚举**：`concept` | `numeric` | `relational` | `composite`

**分类策略**（规则优先 + LLM 兜底）
1. 数值规则：正则匹配数值+单位、数值关键词、标准引用 → `numeric`
2. 关系规则：关系/路径/图关键词 → `relational`
3. 概念规则：定义/机理关键词 → `concept`
4. 复合规则：比较/多意图词 → `composite`
5. v2 增强：intent_hint 加分 + schema_context 加分

### 4.2 路由路径

| 意图 | 检索路径 | 融合策略 |
|------|----------|----------|
| concept | 向量 + 关键词 | RRF 融合 |
| numeric | 关键词 + 数值过滤 | 精确匹配提升 |
| relational | 图遍历 + 子图提取 | BFS + 路径推理 |
| composite | 三路并行 | RRF 融合 + 事实校验 |

### 4.3 v2 上下文增强路由

**route_with_context(ctx)** 流程：
1. Self-RAG 检查：`needs_retrieval=False` → 跳过，返回空结果
2. 意图分类：融合 `intent_hint` + `schema_context`
3. 多查询路由：`len(rewritten) >= 2` → 多路并行 + RRF 融合
4. 自适应参数：使用 `suggested_top_k` / `suggested_depth`

**route_auto(query)** -- 面向 L4 的推荐入口：
- 有 ContextBuilder → build() → route_with_context()
- 无 ContextBuilder → 直接 route()（向后兼容）

### 4.4 RRF 融合

公式：`score(d) = Σ 1/(k + rank_i(d))`，k=60（Cormack et al. 2009）

### 4.5 实体提取

| 类型 | 正则示例 | 匹配 |
|------|----------|------|
| ion | `[A-Z][a-z]?\d*[+-]` | Dy3+, Eu2+ |
| formula | `(?:[A-Z][a-z]?\d*){2,}` | Y2O3, NaYF4 |
| spectral_term | `\d[A-Z]\d+/\d+` | 4F9/2, 5D0 |
| numeric | `\d+\.?\d* (nm|K|eV|...)` | 580nm, 300K |
| keyword | 领域词典匹配 | 跃迁, 猝灭, 机理 |

## 5. 接口设计

### 5.1 ContextBuilder 使用

```python
from dy3_polaris.l3.context_builder import ContextBuilder
from dy3_polaris.l3.api_models import LearnerProfile

builder = ContextBuilder()
ctx = builder.build(
    query="Dy3+的4F9/2能级跃迁波长是多少?",
    learner_profile=profile,      # 可选
    dialog_history=[...],          # 可选
    rewrite_strategies=["expand"], # 可选, None=自动选择
)
# ctx.needs_retrieval → bool
# ctx.active_query → str (指代消解后)
# ctx.intent_hint → str
# ctx.suggested_top_k → int
```

### 5.2 IntentRouter 使用

```python
from dy3_polaris.l3 import IntentRouter, KnowledgeStore
from dy3_polaris.l3.context_builder import ContextBuilder

store = KnowledgeStore()

# v1: 直接路由 (向后兼容)
router = IntentRouter(store)
result = router.route("Dy3+的波长是多少nm?")

# v2: 上下文增强路由
builder = ContextBuilder()
router = IntentRouter(store, context_builder=builder)
ctx = builder.build(query, learner_profile=profile, dialog_history=turns)
result = router.route_with_context(ctx)

# v2: 自动模式 (推荐, L4 入口)
result = router.route_auto(
    query,
    learner_profile=profile,
    dialog_history=turns,
)
```

## 6. 测试覆盖

### 6.1 test_l3_context_builder.py (127 测试)
- ContextBudget: 12 测试
- DialogTurn: 5 测试
- HistoryCompressor (3 策略): 28 测试
- CoreferenceResolver: 15 测试
- SchemaContextInjector: 10 测试
- LearnerContextAdapter: 12 测试
- RetrievalNeedAssessor: 10 测试
- QueryContext: 10 测试
- ContextBuilder 集成: 25 测试

### 6.2 test_l3_intent_router_v2.py (85 测试)
- IntentClassifier v2: 15 测试 (hint/schema 融合)
- RouteWithContext: 13 测试 (Self-RAG/自适应/多查询)
- RouteAuto: 6 测试 (模式选择/兼容性)
- RRF 融合: 6 测试 (去重/参数/聚合)
- 端到端: 11 测试 (ContextBuilder → Router)
- v1 向后兼容: 7 测试
- EntityExtractor: 8 测试
- 边界场景: 8 测试
- 预算与压缩: 6 测试
- Schema 与评估: 5 测试

### 6.3 全量测试

**3905 passed, 0 failed** (18.12s)

## 7. 已知限制

1. **LLM 分类器未接入**：`IntentClassifier._llm_classify()` 和 `LLMClassifier` 协议仅为接口预留
2. **指代消解基于规则**：依赖简单模式匹配（"它" → 前轮实体），复杂指代需 LLM
3. **查询重写基于规则**：5 种策略均为规则+模板实现，无外部 LLM 依赖
4. ** \b 正则边界与 CJK 不兼容**：Python 3 中 CJK 字符属于 \w， \b 在 ASCII 与 CJK 间不触发

## 8. 修复记录

- 运算符优先级修复：`context_builder.py:906` 条件表达式加括号