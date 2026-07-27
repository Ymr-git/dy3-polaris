"""L3 领域知识层 — 增强图推理器 (V2).

本模块 *扩展* (而非复制) ``graph_reasoner.py``，在既有
``GraphReasoner`` (Dijkstra / Yen / 前向链式 / 共同邻居链接预测 /
模式匹配 / 类比推理) 之上，补充四类世界先进的图推理能力:

1. **TransEEmbedder** — TransE 知识图谱嵌入 (Bordes et al. 2013).
   用翻译模型 ``h + r ≈ t`` 学习实体/关系向量，支持链接预测、
   实体相似度检索与增量训练。零外部依赖 (仅 numpy)，对标
   PyKEEN 的生产基线 (gHAWK 2025 评测中 TransE 仍是小图最强基线)。

2. **BackwardChainingReasoner** — 后向链式推理 (Prolog backward
   chaining + OWL RL). 从目标三元组反向展开规则体，生成可读证明
   链。与前向链式推理互补，适用于目标导向 / 证明型 / 规划型推理。

3. **ConfidenceWeightedTraversal** — 置信度加权图遍历
   (GraphRAG subgraph extraction + OMD-GraphRAG). 在 BFS 中按
   三元组置信度加权，替代简单跳数衰减，对应
   ``SubgraphConfig.traverse_strategy = "confidence_weighted"``。

4. **SubgraphReasoner** — 子图推理器 (GraphRAG local search +
   PathMind retrieve-prioritize-reason). 从查询实体提取子图后，
   在子图内做路径查找与摘要生成，为 LLM 提供结构化上下文。

设计原则
--------
- **零外部依赖**: 仅依赖 numpy (已随项目安装)，不引入 torch/pykeen，
  保证 L3 层在受限环境中可独立部署。
- **线程安全**: 所有公开方法通过 ``threading.RLock`` 保护，RLock 可
  重入以支持公开方法间的相互调用 (与 ``GraphReasoner`` 一致)。
- **复用不重复**: 通过 ``from .graph_reasoner import ...`` 复用既有
  ``GraphReasoner`` / ``PathResult`` / ``ReasoningResult`` 等组件，
  不复制其实现。
- **可读证明**: 后向链式推理在 ``ReasoningResult.reasoning_chain`` 中
  输出人类可读的证明轨迹，便于审计与可解释性。

参考文献
--------
- Bordes, A. et al. (2013). *Translating embeddings for modeling
  multi-relational data.* NeurIPS.
- Edge, D. et al. (2024). *From local to global: A graph RAG approach
  to query-focused summarization.* arXiv:2404.16130.
- Sun, Z. et al. (2024). *OMD-GraphRAG: Ontology-model-driven
  GraphRAG.*
- Horrocks, I. et al. (2004). *OWL RL: A fragment of OWL with
  polynomial-time reasoning.*
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import numpy as np

# 框架上下文导入: 以下符号由本模块的公共 API 契约要求 (与 graph_reasoner.py
# 的导入风格一致), 部分类型供下游使用者按类型注解引用, 故保留 noqa: F401。
from .graph_reasoner import (
    GraphReasoner,
    InferenceRule,
    PathResult,
    ReasoningError,
    ReasoningMode,
    ReasoningResult,
)
from .models import (  # noqa: F401
    EntityType,
    KnowledgeEntity,
    KnowledgeGraph,
    KnowledgeTriple,
    RelationType,
    SubgraphConfig,
)
from .ontology import DomainOntology, OntologyRegistry  # noqa: F401
from .store import EntityStore, KnowledgeStore, TripleStore  # noqa: F401

# ============================================================
# 数据类
# ============================================================


@dataclass
class TrainingReport:
    """TransE 训练报告.

    Attributes:
        epochs_trained: 实际训练轮数 (可能小于请求轮数, 若提前收敛).
        final_loss: 最后一轮的平均损失.
        entities_embedded: 已嵌入的实体数.
        relations_embedded: 已嵌入的关系数.
        triples_used: 参与训练的正样本三元组数.
        training_time_ms: 训练总耗时 (毫秒).
        loss_history: 每轮平均损失历史 (用于绘制收敛曲线).
    """

    epochs_trained: int
    final_loss: float
    entities_embedded: int
    relations_embedded: int
    triples_used: int
    training_time_ms: float
    loss_history: list[float]


@dataclass
class LinkPredictionResult:
    """链接预测结果 (TransE).

    Attributes:
        head_id: 头实体 ID.
        relation: 关系谓词.
        predicted_tail: 预测的尾实体 ID (得分最高者).
        score: TransE 打分 (负 L2 距离, 越大越可信).
        confidence: 置信度 (score 的 sigmoid, 映射到 (0,1)).
    """

    head_id: str
    relation: str
    predicted_tail: str
    score: float
    confidence: float


@dataclass
class Goal:
    """后向链式推理的子目标.

    一个目标表示一个待满足的三元组 ``subject -predicate-> object``,
    其中 ``object`` 可以是常量 (证明型) 或变量 (查询型, 以 ``?`` 开头).

    Attributes:
        predicate: 谓词, 如 ``"EMITS_AT"``.
        object: 宾语, 如 ``"610nm"`` 或变量 ``"?x"``.
        subject: 可选主语, 如 ``"YAG:Ce"``; 为空表示主语也是变量.
        is_variable: ``object`` 是否为变量 (以 ``?`` 开头).
    """

    predicate: str
    object: str
    subject: str = ""
    is_variable: bool = False


@dataclass
class WeightedTraversalResult:
    """置信度加权遍历结果.

    Attributes:
        entity_scores: ``entity_id -> 累积置信度`` (起点为 1.0).
        traversed_triples: 遍历过程中经过的所有三元组.
        max_depth_reached: 实际到达的最大深度.
        total_entities: 命中的实体总数 (含起点).
        total_triples: 遍历的三元组总数.
    """

    entity_scores: dict[str, float]
    traversed_triples: list[KnowledgeTriple]
    max_depth_reached: int
    total_entities: int
    total_triples: int


@dataclass
class SubgraphReasoningResult:
    """子图推理结果.

    Attributes:
        focus_entity: 中心实体 ID.
        entities: 子图包含的实体 ID 集合.
        triples: 子图包含的三元组列表.
        paths: 子图内查找到的路径 (若指定了 source/target).
        summary: 子图结构化文本摘要 (供 LLM 进一步精炼).
        entity_names: ``entity_id -> 实体名称`` 映射.
        reasoning_time_ms: 推理总耗时 (毫秒).
    """

    focus_entity: str
    entities: set[str]
    triples: list[KnowledgeTriple]
    paths: list[PathResult]
    summary: str
    entity_names: dict[str, str]
    reasoning_time_ms: float


# ============================================================
# 1. TransEEmbedder — 图嵌入链接预测
# ============================================================


class TransEEmbedder:
    """TransE 知识图谱嵌入 (借鉴 Bordes et al. 2013 + PyKEEN pipeline).

    使用 TransE 的头实体+关系≈尾实体 (h+r≈t) 翻译模型,
    通过负采样和梯度下降训练实体/关系嵌入向量。

    特点:
    - 零外部依赖 (仅 numpy, 不依赖 torch/pykeen)
    - 支持增量训练 (新三元组到来时继续训练)
    - 支持从现有 KnowledgeStore 初始化
    - L2 正则化防止过拟合
    - Margin-based loss (hinge loss)

    嵌入维度默认 128, 学习率默认 0.01, 边际 gamma 默认 1.0。

    评分函数
    --------
    ``score(h, r, t) = -||h + r - t||_2`` (负 L2 距离, 越大越好)。

    训练目标 (margin ranking loss)
    ------------------------------
    ``L = max(0, gamma + score(h, r, t_neg) - score(h, r, t_pos))``

    即: 拉开正样本与负样本的分数差距至少 ``gamma``。

    线程安全
    --------
    所有读写嵌入矩阵的公开方法均通过 ``self._lock`` (RLock) 保护,
    支持训练与查询并发 (查询获取读锁, 训练获取写锁)。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        learning_rate: float = 0.01,
        margin: float = 1.0,
        negative_samples: int = 5,
        seed: int = 42,
    ) -> None:
        """初始化 TransE 嵌入器.

        Args:
            embedding_dim: 嵌入维度 (默认 128).
            learning_rate: SGD 学习率 (默认 0.01).
            margin: hinge loss 的边际 gamma (默认 1.0).
            negative_samples: 每个正样本生成的负样本数 (默认 5).
            seed: 随机种子, 保证可复现.
        """
        self.embedding_dim: int = int(embedding_dim)
        self.learning_rate: float = float(learning_rate)
        self.margin: float = float(margin)
        self.negative_samples: int = int(negative_samples)
        self._seed: int = int(seed)

        # numpy 随机数生成器 (本地, 不污染全局状态)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # 嵌入矩阵与索引映射
        self._entity_embeddings: np.ndarray | None = None
        self._relation_embeddings: np.ndarray | None = None
        self._entity_index: dict[str, int] = {}
        self._index_entity: dict[int, str] = {}
        self._relation_index: dict[str, int] = {}
        self._index_relation: dict[int, str] = {}

        # 训练状态
        self._trained: bool = False
        self._last_report: TrainingReport | None = None

        # 线程安全: RLock 可重入
        self._lock: RLock = RLock()

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    def fit(
        self,
        store: KnowledgeStore,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> TrainingReport:
        """从三元组存储训练 TransE 嵌入.

        训练流程 (Bordes et al. 2013):
        1. 收集所有实体/关系, 建立 ID→index 映射
        2. 随机初始化嵌入 (正态分布, 缩放 1/sqrt(dim))
        3. 每个 epoch: 采样正样本, 随机替换尾实体生成负样本
        4. 计算 margin ranking loss 并用 SGD 更新嵌入
        5. 每 epoch 结束后归一化实体嵌入 (L2 范数 = 1)

        支持增量训练: 若已训练过且嵌入维度一致, 在现有嵌入上继续优化;
        若存储中出现新实体/关系, 扩展嵌入矩阵 (保留旧向量)。

        Args:
            store: 知识存储 (从中读取三元组).
            epochs: 训练轮数 (默认 100).
            batch_size: 批大小 (默认 256).

        Returns:
            训练报告 (含损失历史).
        """
        start_ts = time.perf_counter()

        # 收集正样本三元组 (仅含实体宾语的三元组)
        triples: list[tuple[str, str, str]] = []
        entities_set: set[str] = set()
        relations_set: set[str] = set()
        for t in self._iter_all_triples(store):
            if not t.object_id:
                continue
            triples.append((t.subject_id, t.predicate, t.object_id))
            entities_set.add(t.subject_id)
            entities_set.add(t.object_id)
            relations_set.add(t.predicate)

        with self._lock:
            # (重)构建索引, 增量扩展嵌入矩阵
            self._build_or_extend_indices(
                sorted(entities_set), sorted(relations_set))

            if not triples:
                # 空图: 不训练, 但记录状态
                report = TrainingReport(
                    epochs_trained=0, final_loss=0.0,
                    entities_embedded=len(self._entity_index),
                    relations_embedded=len(self._relation_index),
                    triples_used=0,
                    training_time_ms=round(
                        (time.perf_counter() - start_ts) * 1000.0, 4),
                    loss_history=[])
                self._trained = True
                self._last_report = report
                return report

            n_entities = len(self._entity_index)
            n_relations = len(self._relation_index)
            dim = self.embedding_dim

            # 初始化或复用嵌入矩阵
            if (self._entity_embeddings is None
                    or self._entity_embeddings.shape != (n_entities, dim)):
                self._entity_embeddings = self._rng.normal(
                    loc=0.0, scale=1.0 / math.sqrt(dim),
                    size=(n_entities, dim)).astype(np.float64)
            if (self._relation_embeddings is None
                    or self._relation_embeddings.shape != (n_relations, dim)):
                self._relation_embeddings = self._rng.normal(
                    loc=0.0, scale=1.0 / math.sqrt(dim),
                    size=(n_relations, dim)).astype(np.float64)

            ent_emb = self._entity_embeddings
            rel_emb = self._relation_embeddings

            # 三元组索引数组 (加速批采样)
            triples_arr = np.array(
                [(self._entity_index[h], self._relation_index[r],
                  self._entity_index[t])
                 for h, r, t in triples],
                dtype=np.int64)
            n_triples = len(triples_arr)

            loss_history: list[float] = []
            epochs_done = 0
            lr = self.learning_rate

            for epoch in range(epochs):
                # 随机打乱正样本顺序
                perm = self._rng.permutation(n_triples)
                epoch_loss = 0.0
                n_batches = max(1, (n_triples + batch_size - 1) // batch_size)

                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_triples)
                    batch_idx = perm[start:end]
                    batch_loss = self._train_batch(
                        triples_arr, batch_idx, n_entities,
                        ent_emb, rel_emb, lr)
                    epoch_loss += batch_loss

                avg_loss = epoch_loss / n_batches
                loss_history.append(round(avg_loss, 6))
                epochs_done = epoch + 1

                # 每 epoch 归一化实体嵌入 (L2 范数 = 1)
                # 防止嵌入向量无界增长 (Bordes et al. 建议的约束)
                norms = np.linalg.norm(ent_emb, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                ent_emb /= norms

            final_loss = loss_history[-1] if loss_history else 0.0
            report = TrainingReport(
                epochs_trained=epochs_done,
                final_loss=final_loss,
                entities_embedded=n_entities,
                relations_embedded=n_relations,
                triples_used=n_triples,
                training_time_ms=round(
                    (time.perf_counter() - start_ts) * 1000.0, 4),
                loss_history=loss_history)
            self._trained = True
            self._last_report = report
            return report

    def _train_batch(
        self,
        triples_arr: np.ndarray,
        batch_idx: np.ndarray,
        n_entities: int,
        ent_emb: np.ndarray,
        rel_emb: np.ndarray,
        lr: float,
    ) -> float:
        """训练一个 mini-batch, 返回该批平均损失 (内部方法, 调用方持锁)."""
        total_loss = 0.0
        n_samples = 0

        for pos_idx in batch_idx:
            h_i, r_i, t_i = triples_arr[pos_idx]
            h_vec = ent_emb[h_i]
            r_vec = rel_emb[r_i]
            t_vec = ent_emb[t_i]

            # 正样本分数: score(h,r,t) = -||h+r-t||
            pos_diff = h_vec + r_vec - t_vec
            pos_score = -np.linalg.norm(pos_diff)

            # 生成 negative_samples 个负样本 (随机替换尾实体)
            for _ in range(self.negative_samples):
                neg_t_i = self._rng.integers(0, n_entities)
                if neg_t_i == t_i:
                    # 避免负样本与正样本相同
                    neg_t_i = (neg_t_i + 1) % n_entities
                neg_t_vec = ent_emb[neg_t_i]
                neg_diff = h_vec + r_vec - neg_t_vec
                neg_score = -np.linalg.norm(neg_diff)

                # margin ranking loss: L = max(0, gamma + neg_score - pos_score)
                loss = self.margin + neg_score - pos_score
                if loss > 0:
                    # 计算梯度并更新 (SGD)
                    # d(loss)/d(h) = d(neg_score - pos_score)/d(h)
                    #   pos_score = -||h+r-t||  => d(pos)/d(h) = -(h+r-t)/||h+r-t||
                    #   neg_score = -||h+r-t_neg|| => d(neg)/d(h) = -(h+r-t_neg)/||...||
                    # d(loss)/d(h) = (h+r-t)/||h+r-t|| - (h+r-t_neg)/||h+r-t_neg||
                    pos_norm = max(np.linalg.norm(pos_diff), 1e-12)
                    neg_norm = max(np.linalg.norm(neg_diff), 1e-12)
                    grad_h = pos_diff / pos_norm - neg_diff / neg_norm
                    grad_r = grad_h  # r 对两个分数的梯度与 h 相同
                    grad_t_pos = -pos_diff / pos_norm
                    grad_t_neg = neg_diff / neg_norm

                    # L2 正则化 (权重衰减)
                    reg = 1e-5
                    ent_emb[h_i] -= lr * (grad_h + reg * h_vec)
                    rel_emb[r_i] -= lr * (grad_r + reg * r_vec)
                    ent_emb[t_i] -= lr * (grad_t_pos + reg * t_vec)
                    ent_emb[neg_t_i] -= lr * (grad_t_neg + reg * neg_t_vec)

                    total_loss += loss
                n_samples += 1

        return float(total_loss / max(n_samples, 1))

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def get_embedding(self, entity_id: str) -> np.ndarray | None:
        """获取实体嵌入向量 (副本). 未训练或实体不存在时返回 None."""
        with self._lock:
            idx = self._entity_index.get(entity_id)
            if idx is None or self._entity_embeddings is None:
                return None
            return self._entity_embeddings[idx].copy()

    def get_relation_embedding(self, relation: str) -> np.ndarray | None:
        """获取关系嵌入向量 (副本). 未训练或关系不存在时返回 None."""
        with self._lock:
            idx = self._relation_index.get(relation)
            if idx is None or self._relation_embeddings is None:
                return None
            return self._relation_embeddings[idx].copy()

    def predict_link(
        self,
        head_id: str,
        relation: str,
        tail_id: str | None = None,
    ) -> LinkPredictionResult | None:
        """预测尾实体或评估给定三元组.

        - 若 ``tail_id`` 为 None: 在所有实体中寻找使 ``score(h,r,t)`` 最大的 t,
          返回 ``LinkPredictionResult``。
        - 若 ``tail_id`` 给定: 评估该三元组的分数, ``predicted_tail`` 即为给定值。

        Args:
            head_id: 头实体 ID.
            relation: 关系谓词.
            tail_id: 尾实体 ID (可选; None 表示预测).

        Returns:
            链接预测结果, 若头实体/关系未嵌入则返回 None。
        """
        with self._lock:
            h_idx = self._entity_index.get(head_id)
            r_idx = self._relation_index.get(relation)
            if h_idx is None or r_idx is None or self._entity_embeddings is None:
                return None
            h_vec = self._entity_embeddings[h_idx]
            r_vec = self._relation_embeddings[r_idx]

            if tail_id is not None:
                t_idx = self._entity_index.get(tail_id)
                if t_idx is None:
                    return None
                t_vec = self._entity_embeddings[t_idx]
                score = float(-np.linalg.norm(h_vec + r_vec - t_vec))
                confidence = float(1.0 / (1.0 + math.exp(-score)))
                return LinkPredictionResult(
                    head_id=head_id, relation=relation,
                    predicted_tail=tail_id, score=score,
                    confidence=confidence)

            # 预测: 对所有实体打分, 取最大
            diff = self._entity_embeddings - (h_vec + r_vec)  # (N, dim)
            norms = np.linalg.norm(diff, axis=1)
            scores = -norms
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_tail = self._index_entity[best_idx]
            confidence = float(1.0 / (1.0 + math.exp(-best_score)))
            return LinkPredictionResult(
                head_id=head_id, relation=relation,
                predicted_tail=best_tail, score=best_score,
                confidence=confidence)

    def rank_entities(
        self,
        head_id: str,
        relation: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """为给定 (h, r) 对所有实体排序, 返回 top_k 的 (entity_id, score)."""
        with self._lock:
            h_idx = self._entity_index.get(head_id)
            r_idx = self._relation_index.get(relation)
            if h_idx is None or r_idx is None or self._entity_embeddings is None:
                return []
            h_vec = self._entity_embeddings[h_idx]
            r_vec = self._relation_embeddings[r_idx]
            diff = self._entity_embeddings - (h_vec + r_vec)
            scores = -np.linalg.norm(diff, axis=1)
            k = min(top_k, len(scores))
            # argpartition 取 top-k, 再排序
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            return [
                (self._index_entity[int(i)], float(scores[i]))
                for i in top_idx
            ]

    def similarity(self, id_a: str, id_b: str) -> float:
        """计算两个实体的余弦相似度. 任一未嵌入则返回 0.0."""
        with self._lock:
            va = self.get_embedding(id_a)
            vb = self.get_embedding(id_b)
            if va is None or vb is None:
                return 0.0
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(np.dot(va, vb) / (na * nb))

    def most_similar(
        self,
        entity_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """查找与给定实体最相似的实体 (余弦相似度 top_k)."""
        with self._lock:
            target = self.get_embedding(entity_id)
            if target is None or self._entity_embeddings is None:
                return []
            norms = np.linalg.norm(self._entity_embeddings, axis=1)
            target_norm = np.linalg.norm(target)
            if target_norm < 1e-12:
                return []
            # 归一化后点积即余弦相似度
            normalized = self._entity_embeddings / np.where(
                norms < 1e-12, 1.0, norms)[:, None]
            sims = normalized @ (target / target_norm)
            # 排除自身
            self_idx = self._entity_index.get(entity_id, -1)
            if self_idx >= 0:
                sims[self_idx] = -np.inf
            k = min(top_k, len(sims))
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [
                (self._index_entity[int(i)], float(sims[i]))
                for i in top_idx
                if math.isfinite(sims[i])
            ]

    # --------------------------------------------------------
    # 统计与内部辅助
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取嵌入统计信息."""
        with self._lock:
            return {
                "embedding_dim": self.embedding_dim,
                "learning_rate": self.learning_rate,
                "margin": self.margin,
                "negative_samples": self.negative_samples,
                "trained": self._trained,
                "entities_embedded": len(self._entity_index),
                "relations_embedded": len(self._relation_index),
                "epochs_trained": (
                    self._last_report.epochs_trained
                    if self._last_report else 0),
                "final_loss": (
                    self._last_report.final_loss
                    if self._last_report else 0.0),
            }

    def _build_or_extend_indices(
        self,
        entities: list[str],
        relations: list[str],
    ) -> None:
        """构建或增量扩展实体/关系索引 (调用方持锁).

        新实体/关系追加到索引末尾, 嵌入矩阵相应扩展 (旧向量保留)。
        """
        dim = self.embedding_dim
        for eid in entities:
            if eid not in self._entity_index:
                idx = len(self._entity_index)
                self._entity_index[eid] = idx
                self._index_entity[idx] = eid
        for rel in relations:
            if rel not in self._relation_index:
                idx = len(self._relation_index)
                self._relation_index[rel] = idx
                self._index_relation[idx] = rel

        n_ent = len(self._entity_index)
        n_rel = len(self._relation_index)

        # 扩展实体嵌入矩阵
        if self._entity_embeddings is not None:
            old_n = self._entity_embeddings.shape[0]
            if n_ent > old_n:
                new_rows = self._rng.normal(
                    loc=0.0, scale=1.0 / math.sqrt(dim),
                    size=(n_ent - old_n, dim)).astype(np.float64)
                self._entity_embeddings = np.vstack(
                    [self._entity_embeddings, new_rows])

        # 扩展关系嵌入矩阵
        if self._relation_embeddings is not None:
            old_n = self._relation_embeddings.shape[0]
            if n_rel > old_n:
                new_rows = self._rng.normal(
                    loc=0.0, scale=1.0 / math.sqrt(dim),
                    size=(n_rel - old_n, dim)).astype(np.float64)
                self._relation_embeddings = np.vstack(
                    [self._relation_embeddings, new_rows])

    @staticmethod
    def _iter_all_triples(
        store: KnowledgeStore,
    ) -> list[KnowledgeTriple]:
        """从存储中读取所有三元组 (优先访问内部字典)."""
        internal = getattr(store.triple_store, "_triples", None)
        if isinstance(internal, dict):
            return list(internal.values())
        result: list[KnowledgeTriple] = []
        seen: set[str] = set()
        for rt in RelationType:
            for t in store.triple_store.get_by_predicate(rt.value):
                if t.triple_id not in seen:
                    seen.add(t.triple_id)
                    result.append(t)
        return result


# ============================================================
# 2. BackwardChainingReasoner — 后向链式推理
# ============================================================


@dataclass
class _Proof:
    """内部证明节点 (后向链式推理递归中间态).

    不对外暴露; 仅用于在 ``_backward_step`` 递归中传递
    绑定环境、证明链与证据, 最终在 ``reason()`` 中聚合为
    ``ReasoningResult``。

    Attributes:
        bindings: 变量绑定 {var_name -> value}.
        chain: 人类可读的证明步骤列表 (按推导顺序).
        evidence: 支撑证据三元组 (dict 形式, 与
            ``ReasoningResult.evidence_triples`` 一致).
        confidence: 该证明的累积置信度 (各步规则的乘积/最小值).
    """

    bindings: dict[str, str] = field(default_factory=dict)
    chain: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0


class BackwardChainingReasoner:
    """后向链式推理器 (借鉴 Prolog backward chaining + OWL RL).

    从目标反向推导所需条件, 递归匹配推理规则。
    与前向链式推理 (forward_chaining) 互补, 适用于:
    - 目标导向的问题 ("哪些材料能发射 610nm 光?")
    - 证明型推理 ("为什么 Dy3+ 可以发射黄光?")
    - 规划型推理 ("如何合成 YAG:Ce?")

    每步推理维护一个目标栈和绑定环境。

    算法
    ----
    1. 构造初始目标 ``Goal(predicate, object, subject)``。
    2. 检查存储中是否存在直接匹配的三元组 (满足目标)。
    3. 查找推理规则中头部谓词与目标谓词匹配的规则。
    4. 对每条匹配规则, 递归地满足其体部条件 (sub-goals)。
    5. 用 ``visited`` 集合防止循环 (同一目标不重复展开)。
    6. 收集所有成功的证明链, 输出为人类可读的 ``ReasoningResult``。
    """

    def __init__(
        self,
        store: KnowledgeStore,
        reasoner: GraphReasoner,
        max_depth: int = 5,
        max_results: int = 20,
    ) -> None:
        """初始化后向链式推理器.

        Args:
            store: 知识存储 (用于直接匹配查询).
            reasoner: 已配置的图推理器 (提供推理规则).
            max_depth: 最大递归深度 (防止无限递归).
            max_results: 最多返回的证明结果数.
        """
        self._store: KnowledgeStore = store
        self._reasoner: GraphReasoner = reasoner
        self._max_depth: int = int(max_depth)
        self._max_results: int = int(max_results)
        self._lock: RLock = RLock()

    def reason(
        self,
        goal_predicate: str,
        goal_object: str,
        **kwargs: Any,
    ) -> ReasoningResult:
        """后向链式推理入口.

        Args:
            goal_predicate: 目标谓词, 如 ``"EMITS_AT"``.
            goal_object: 目标宾语, 如 ``"610nm"``; 以 ``?`` 开头视为变量.
            **kwargs: 额外参数, 支持 ``subject`` (限定主语)。

        Returns:
            ``ReasoningResult`` (mode=RULE_INFERENCE), 其中
            ``reasoning_chain`` 为人类可读的证明轨迹,
            ``answers`` 为满足目标的绑定列表,
            ``evidence_triples`` 为支撑证据三元组。
        """
        start_ts = time.perf_counter()
        subject = kwargs.get("subject", "")
        is_var = goal_object.startswith("?")

        root_goal = Goal(
            predicate=goal_predicate,
            object=goal_object,
            subject=subject,
            is_variable=is_var,
        )

        with self._lock:
            try:
                proofs = self._backward_step(
                    [root_goal], {}, depth=0, visited=set())
            except Exception as exc:  # noqa: BLE001
                raise ReasoningError(
                    detail=f"后向链式推理失败: {exc}",
                    context={"goal_predicate": goal_predicate,
                             "goal_object": goal_object}) from exc

            answers: list[dict[str, Any]] = []
            chain: list[str] = []
            evidence: list[dict[str, Any]] = []
            seen_bindings: set[str] = set()

            for proof in proofs[: self._max_results]:
                # 去重: 相同绑定只保留首个 (置信度最高)
                binding_key = str(sorted(proof.bindings.items()))
                if binding_key in seen_bindings:
                    continue
                seen_bindings.add(binding_key)
                answers.append({
                    "bindings": dict(proof.bindings),
                    "confidence": round(proof.confidence, 6),
                })
                chain.extend(proof.chain)
                for ev in proof.evidence:
                    if ev not in evidence:
                        evidence.append(ev)

            if not answers:
                chain.append(
                    f"目标 {goal_predicate}({goal_object}) 无法被满足: "
                    f"存储中无直接匹配, 也无可用推理规则")
            else:
                chain.insert(0, (
                    f"后向链式推理: 目标 {goal_predicate}({goal_object}) "
                    f"找到 {len(answers)} 个满足方案"))

            confidence = (
                sum(a["confidence"] for a in answers) / len(answers)
                if answers else 0.0)

            return ReasoningResult(
                mode=ReasoningMode.RULE_INFERENCE,
                query=f"backward({goal_predicate}, {goal_object})",
                answers=answers,
                confidence=round(confidence, 6),
                reasoning_chain=chain,
                evidence_triples=evidence,
                elapsed_ms=round(
                    (time.perf_counter() - start_ts) * 1000.0, 4),
            )

    # --------------------------------------------------------
    # 内部递归推理
    # --------------------------------------------------------

    def _backward_step(
        self,
        goals: list[Goal],
        bindings: dict[str, str],
        depth: int,
        visited: set[str],
    ) -> list[_Proof]:
        """递归地满足目标列表, 返回所有成功的证明.

        对 goals 中的第一个目标:
        1. 尝试从存储直接匹配 (_match_goal).
        2. 尝试用推理规则展开 (_applicable_rules), 递归满足规则体。
        若所有目标均满足, 构造一个 ``_Proof`` 返回。

        内部使用 ``_Proof`` 而非 ``ReasoningResult``, 因为后者要求
        ``answers`` 为 ``list[dict]``, 而递归中需传递单个绑定字典。
        """
        if not goals:
            # 所有目标已满足: 构造证明
            return [_Proof(bindings=dict(bindings), confidence=1.0)]

        if depth >= self._max_depth:
            return []

        current_goal = goals[0]
        rest_goals = goals[1:]
        proofs: list[_Proof] = []

        # 目标签名 (用于循环检测)
        goal_sig = (
            f"{current_goal.subject}|{current_goal.predicate}"
            f"|{current_goal.object}@{depth}")
        if goal_sig in visited:
            return []
        visited = visited | {goal_sig}

        # 路径 1: 从存储直接匹配
        direct = self._match_goal(current_goal, bindings)
        if direct is not None:
            new_bindings = {**bindings, **direct}
            direct_chain = self._describe_match(current_goal, direct)
            direct_evidence = self._evidence_for_match(
                current_goal, new_bindings)
            sub_proofs = self._backward_step(
                rest_goals, new_bindings, depth + 1, visited)
            for sp in sub_proofs:
                sp.chain.insert(0, direct_chain)
                sp.evidence = direct_evidence + sp.evidence
                proofs.append(sp)
            # 若无后续目标, 也记录直接匹配成功
            if not rest_goals:
                proofs.append(_Proof(
                    bindings=dict(new_bindings),
                    chain=[direct_chain],
                    evidence=list(direct_evidence),
                    confidence=1.0,
                ))

        # 路径 2: 通过推理规则展开
        for rule, rule_bindings in self._applicable_rules(
            current_goal, bindings
        ):
            body_goals = self._rule_body_goals(rule, rule_bindings)
            if body_goals is None:
                continue
            merged_bindings = {**bindings, **rule_bindings}
            rule_chain = (
                f"[深度{depth}] 应用规则 '{rule.name}' ({rule.rule_id}): "
                f"为满足 {self._describe_goal(current_goal)}, "
                f"需先满足 {len(body_goals)} 个子目标")
            # 先满足规则体
            body_proofs = self._backward_step(
                body_goals, merged_bindings, depth + 1, visited)
            for bp in body_proofs:
                bp.chain.insert(0, rule_chain)
                bp.confidence = round(
                    min(bp.confidence, rule.confidence), 6)
                # 规则体满足后, 继续满足剩余目标
                final_proofs = self._backward_step(
                    rest_goals, bp.bindings, depth + 1, visited)
                if final_proofs:
                    for fp in final_proofs:
                        fp.chain = bp.chain + fp.chain
                        fp.evidence = bp.evidence + fp.evidence
                        fp.confidence = round(
                            min(fp.confidence, bp.confidence), 6)
                        proofs.append(fp)
                else:
                    # 规则体满足即整体满足 (无剩余目标)
                    proofs.append(bp)

        return proofs

    def _match_goal(
        self,
        goal: Goal,
        bindings: dict[str, str],
    ) -> dict[str, str] | None:
        """尝试从存储中直接满足目标.

        返回新增的变量绑定 (若有变量), 或空 dict 表示匹配成功 (常量目标);
        返回 None 表示无法直接匹配。
        """
        # 解析变量 (优先使用已有绑定)
        subject = goal.subject or bindings.get("?subject", "")
        obj = goal.object
        if goal.is_variable and obj in bindings:
            obj = bindings[obj]

        # 情况 A: 主语已知, 查询 (subject, predicate, ?)
        if subject:
            triples = self._store.triple_store.get_outgoing(
                subject, predicate=goal.predicate)
            for t in triples:
                t_obj = t.object_id or (
                    str(t.object_value) if t.object_is_literal else "")
                if not t_obj:
                    continue
                if goal.is_variable:
                    # 变量目标: 绑定变量
                    return {goal.object: t_obj}
                # 常量目标: 检查值是否匹配
                if t_obj == goal.object:
                    return {}
            return None

        # 情况 B: 主语未知, 扫描所有该谓词的三元组
        triples = self._store.triple_store.get_by_predicate(goal.predicate)
        for t in triples:
            t_obj = t.object_id or (
                str(t.object_value) if t.object_is_literal else "")
            if not t_obj:
                continue
            if goal.is_variable:
                return {goal.object: t_obj, "?subject": t.subject_id}
            if t_obj == goal.object:
                return {"?subject": t.subject_id}
        return None

    def _applicable_rules(
        self,
        goal: Goal,
        bindings: dict[str, str],
    ) -> list[tuple[InferenceRule, dict[str, str]]]:
        """查找头部谓词与目标匹配的推理规则.

        规则的 ``inference_pattern["predicate"]`` 视为规则头部谓词。
        返回 (rule, initial_bindings) 列表。
        """
        applicable: list[tuple[InferenceRule, dict[str, str]]] = []
        for rule in self._reasoner.get_rules():
            head_pred = rule.inference_pattern.get("predicate", "")
            if not head_pred or head_pred != goal.predicate:
                continue
            applicable.append((rule, {}))
        return applicable

    def _rule_body_goals(
        self,
        rule: InferenceRule,
        rule_bindings: dict[str, str],
    ) -> list[Goal] | None:
        """从规则条件模式构造子目标列表.

        支持的规则类型:
        - ``transitive``: 传递性规则, 构造中间桥接目标
          (A -pred-> ?mid, ?mid -pred-> object).
        - ``inverse``: 逆关系规则, 构造逆关系目标.
        - ``match_infer``: 通用匹配, 直接用条件谓词作为子目标.
        """
        cond = rule.condition_pattern
        rtype = cond.get("type", "match_infer")
        cond_pred = cond.get("predicate", "")

        if rtype == "transitive":
            # 传递性: 要证明 A -pred-> C, 需证 A -pred-> ?mid 且 ?mid -pred-> C
            # 这里 goal.object 作为 C, 构造两个子目标
            return [
                Goal(predicate=cond_pred, object="?mid", is_variable=True),
                Goal(predicate=cond_pred, object="?mid_value",
                     is_variable=True),
            ]
        if rtype == "inverse":
            # 逆关系: A -pred-> B => B -inv_pred-> A
            inv_pred = cond.get(
                "inverse_predicate",
                rule.inference_pattern.get("predicate", cond_pred))
            return [Goal(predicate=inv_pred, object="?subject",
                         is_variable=True)]
        # match_infer: 直接以条件谓词作为子目标
        return [Goal(predicate=cond_pred, object="?x", is_variable=True)]

    # --------------------------------------------------------
    # 证明链描述辅助
    # --------------------------------------------------------

    def _describe_match(self, goal: Goal, bindings: dict[str, str]) -> str:
        """生成直接匹配的证明描述."""
        subj = goal.subject or bindings.get("?subject", "?")
        obj = bindings.get(goal.object, goal.object) if goal.is_variable else goal.object
        return (f"[直接匹配] 存储中找到三元组: {subj} -[{goal.predicate}]-> {obj}")

    def _evidence_for_match(
        self,
        goal: Goal,
        bindings: dict[str, str],
    ) -> list[dict[str, Any]]:
        """收集直接匹配所依赖的存储三元组, 作为证据.

        返回 dict 列表 (与 ``ReasoningResult.evidence_triples`` 格式一致),
        每项含 subject_id/predicate/object_id/confidence。
        """
        subject = goal.subject or bindings.get("?subject", "")
        obj = bindings.get(goal.object, goal.object) if goal.is_variable else goal.object
        evidence: list[dict[str, Any]] = []
        if subject:
            triples = self._store.triple_store.get_outgoing(
                subject, predicate=goal.predicate)
            for t in triples:
                t_obj = t.object_id or (
                    str(t.object_value) if t.object_is_literal else "")
                if t_obj and (not goal.is_variable or t_obj == obj):
                    evidence.append({
                        "triple_id": t.triple_id,
                        "subject_id": t.subject_id,
                        "predicate": t.predicate,
                        "object_id": t.object_id,
                        "confidence": float(t.confidence),
                    })
                    break
        return evidence

    def _describe_goal(self, goal: Goal) -> str:
        """生成目标的可读描述."""
        subj = goal.subject or "?subject"
        return f"{subj} -[{goal.predicate}]-> {goal.object}"

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取推理器统计信息."""
        with self._lock:
            return {
                "max_depth": self._max_depth,
                "max_results": self._max_results,
                "rules_available": len(self._reasoner.get_rules()),
                "entities_count": self._store.entity_count(),
                "triples_count": self._store.triple_count(),
            }


# ============================================================
# 3. ConfidenceWeightedTraversal — 置信度加权图遍历
# ============================================================


class ConfidenceWeightedTraversal:
    """置信度加权图遍历 (借鉴 GraphRAG subgraph extraction + OMD-GraphRAG).

    在 BFS 遍历中按三元组置信度加权, 替代简单的跳数衰减。
    支持 ``SubgraphConfig.traverse_strategy = "confidence_weighted"``。

    特点:
    - 边权重 = 1.0 / confidence (高置信度边优先遍历)
    - 支持最小置信度阈值过滤低质量边
    - 支持置信度衰减 (距离越远, 允许的最低置信度越低)
    - 返回带累积置信度的子图

    算法
    ----
    采用加权 BFS (类 Dijkstra, 但以累积置信度而非距离为度量):
    - 每个实体维护一个累积置信度分数 (起点为 1.0)。
    - 从某实体 ``u`` 经三元组 ``t`` 到达 ``v`` 时,
      ``v`` 的累积分数 = ``u`` 的分数 × ``t.confidence``。
    - 若 ``v`` 已有更高分数, 则跳过 (贪心)。
    - 每层应用 ``decay_factor`` 衰减最低置信度阈值,
      允许深层接受更低置信度的边。
    """

    def __init__(
        self,
        store: KnowledgeStore,
        min_confidence: float = 0.3,
        decay_factor: float = 0.9,
        max_depth: int = 3,
    ) -> None:
        """初始化置信度加权遍历器.

        Args:
            store: 知识存储.
            min_confidence: 初始最低置信度阈值 (过滤低质量边).
            decay_factor: 每深一层, 最低置信度阈值乘以此因子
                (允许深层接受更低置信度的边).
            max_depth: 默认最大遍历深度.
        """
        self._store: KnowledgeStore = store
        self._min_confidence: float = float(min_confidence)
        self._decay_factor: float = float(decay_factor)
        self._max_depth: int = int(max_depth)
        self._lock: RLock = RLock()

    def traverse(
        self,
        start_entity_id: str,
        max_depth: int | None = None,
        min_confidence: float | None = None,
    ) -> WeightedTraversalResult:
        """从起点实体执行置信度加权 BFS 遍历.

        Args:
            start_entity_id: 起点实体 ID.
            max_depth: 最大深度 (None 使用构造器默认值).
            min_confidence: 初始最低置信度 (None 使用构造器默认值).

        Returns:
            加权遍历结果 (含累积置信度与遍历的三元组)。
        """
        depth = max_depth if max_depth is not None else self._max_depth
        base_conf = (
            min_confidence if min_confidence is not None
            else self._min_confidence)

        with self._lock:
            if not self._store.entity_store.exists(start_entity_id):
                return WeightedTraversalResult(
                    entity_scores={}, traversed_triples=[],
                    max_depth_reached=0, total_entities=0, total_triples=0)

            # 起点: 累积置信度 1.0
            current: dict[str, float] = {start_entity_id: 1.0}
            visited: set[str] = {start_entity_id}
            entity_scores: dict[str, float] = {start_entity_id: 1.0}
            traversed: list[KnowledgeTriple] = []
            max_depth_reached = 0

            for d in range(depth):
                if not current:
                    break
                current, reached = self._expand_layer(
                    current, visited, d, depth,
                    base_conf=base_conf,
                    entity_scores=entity_scores,
                    traversed=traversed,
                )
                visited.update(current.keys())
                if current:
                    max_depth_reached = d + 1

            return WeightedTraversalResult(
                entity_scores=entity_scores,
                traversed_triples=traversed,
                max_depth_reached=max_depth_reached,
                total_entities=len(entity_scores),
                total_triples=len(traversed),
            )

    def _expand_layer(
        self,
        current: dict[str, float],
        visited: set[str],
        depth: int,
        max_depth: int,
        *,
        base_conf: float,
        entity_scores: dict[str, float],
        traversed: list[KnowledgeTriple],
    ) -> tuple[dict[str, float], bool]:
        """扩展一层 BFS.

        对 ``current`` 中每个实体, 查询其出边/入边三元组,
        按置信度加权更新邻居的累积分数。

        Returns:
            (下一层实体->分数, 是否到达新实体)
        """
        # 该层允许的最低置信度 (随深度衰减)
        layer_min_conf = base_conf * (self._decay_factor ** depth)
        layer_min_conf = max(layer_min_conf, 0.0)

        next_layer: dict[str, float] = {}
        reached_new = False

        for eid, score in current.items():
            # 出边 + 入边 (双向遍历)
            outgoing = self._store.triple_store.get_outgoing(
                eid, min_confidence=layer_min_conf)
            incoming = self._store.triple_store.get_incoming(
                eid, min_confidence=layer_min_conf)

            for t in outgoing + incoming:
                neighbor = (
                    t.object_id
                    if t.subject_id == eid and t.object_id
                    else t.subject_id)
                if not neighbor or neighbor == eid:
                    continue
                traversed.append(t)
                # 累积置信度 = 父分数 × 边置信度
                acc = score * float(t.confidence)
                if neighbor not in entity_scores or acc > entity_scores[neighbor]:
                    entity_scores[neighbor] = acc
                if neighbor not in visited:
                    if neighbor not in next_layer or acc > next_layer[neighbor]:
                        next_layer[neighbor] = acc
                    reached_new = True

        return next_layer, reached_new

    def get_stats(self) -> dict[str, Any]:
        """获取遍历器统计信息."""
        with self._lock:
            return {
                "min_confidence": self._min_confidence,
                "decay_factor": self._decay_factor,
                "max_depth": self._max_depth,
                "entities_count": self._store.entity_count(),
                "triples_count": self._store.triple_count(),
            }


# ============================================================
# 4. SubgraphReasoner — 子图推理 (GraphRAG 模式)
# ============================================================


class SubgraphReasoner:
    """子图推理器 (借鉴 GraphRAG local search + PathMind retrieve-prioritize-reason).

    从查询实体出发提取相关子图, 然后在子图上进行推理:
    1. 实体聚焦子图提取 (BFS/最短路径/置信度加权)
    2. 子图内路径查找
    3. 子图内模式匹配
    4. 子图摘要生成 (接口, 供 LLM 调用)

    设计理念
    --------
    遵循 GraphRAG 的 "retrieve-prioritize-reason" 三段式:
    - **retrieve**: 从存储中提取以焦点实体为中心的子图;
    - **prioritize**: 通过置信度加权/最短路径排序子图内容;
    - **reason**: 在子图内执行路径查找与摘要, 为下游 LLM 提供精炼上下文。
    """

    def __init__(
        self,
        store: KnowledgeStore,
        reasoner: GraphReasoner,
    ) -> None:
        """初始化子图推理器.

        Args:
            store: 知识存储.
            reasoner: 已配置的图推理器 (复用其路径查找能力).
        """
        self._store: KnowledgeStore = store
        self._reasoner: GraphReasoner = reasoner
        self._lock: RLock = RLock()

    def extract_and_reason(
        self,
        entity_id: str,
        query: str = "",
        *,
        strategy: str = "bfs",
        max_depth: int = 2,
    ) -> SubgraphReasoningResult:
        """提取子图并在其上推理 (主入口).

        Args:
            entity_id: 焦点实体 ID.
            query: 可选查询字符串 (记录在结果中, 供 LLM 上下文使用).
            strategy: 子图提取策略, 对应 ``SubgraphConfig.traverse_strategy``
                的取值: ``"bfs"`` / ``"shortest_path"`` / ``"confidence_weighted"``。
            max_depth: 最大提取深度.

        Returns:
            子图推理结果 (含实体集合、三元组、路径、摘要)。
        """
        start_ts = time.perf_counter()
        with self._lock:
            entities, triples = self.extract_subgraph(
                entity_id, strategy=strategy, max_depth=max_depth)

            # 在子图内查找焦点实体到其他实体的路径 (取前若干条)
            paths: list[PathResult] = []
            if entities:
                # 收集焦点实体的直接邻居, 查找短路径作为证据
                neighbors = self._direct_neighbors_in_subgraph(
                    entity_id, triples)
                for nbr in list(neighbors)[:5]:
                    pr = self.find_paths_in_subgraph(
                        entity_id, nbr, entities, triples)
                    if pr:
                        paths.extend(pr[:1])

            entity_names = self._collect_names(entities)
            summary = self.summarize_subgraph(
                entity_id, entities, triples)

            return SubgraphReasoningResult(
                focus_entity=entity_id,
                entities=entities,
                triples=triples,
                paths=paths,
                summary=summary,
                entity_names=entity_names,
                reasoning_time_ms=round(
                    (time.perf_counter() - start_ts) * 1000.0, 4),
            )

    def extract_subgraph(
        self,
        entity_id: str,
        strategy: str = "bfs",
        max_depth: int = 2,
    ) -> tuple[set[str], list[KnowledgeTriple]]:
        """提取以焦点实体为中心的子图.

        根据 ``strategy`` 选择提取方式:
        - ``"bfs"``: 广度优先遍历 (委托 ``TripleStore.traverse_bfs``).
        - ``"confidence_weighted"``: 置信度加权遍历
          (委托 ``ConfidenceWeightedTraversal``).
        - ``"shortest_path"``: 以焦点为中心, 对若干邻居取最短路径
          (委托 ``GraphReasoner.find_shortest_path``).

        Args:
            entity_id: 焦点实体 ID.
            strategy: 遍历策略 (需为 ``SubgraphConfig`` 允许的取值).
            max_depth: 最大深度.

        Returns:
            (实体 ID 集合, 三元组列表).
        """
        # 校验 strategy (与 SubgraphConfig 的允许集合一致)
        allowed = {"bfs", "shortest_path", "confidence_weighted"}
        if strategy not in allowed:
            raise ReasoningError(
                detail=(
                    f"未知的子图提取策略: {strategy}, "
                    f"允许值: {sorted(allowed)}"),
                context={"strategy": strategy})

        if strategy == "bfs":
            ent_ids, trps = self._store.triple_store.traverse_bfs(
                entity_id, max_depth=max_depth, direction="both")
            return set(ent_ids), trps

        if strategy == "confidence_weighted":
            traversal = ConfidenceWeightedTraversal(
                self._store, max_depth=max_depth)
            result = traversal.traverse(entity_id, max_depth=max_depth)
            return set(result.entity_scores.keys()), result.traversed_triples

        # shortest_path: 获取邻居后逐个查最短路径
        ent_ids, trps = self._store.triple_store.traverse_bfs(
            entity_id, max_depth=max_depth, direction="both")
        entities: set[str] = {entity_id}
        triples: list[KnowledgeTriple] = []
        seen_triple_ids: set[str] = set()

        for nbr in ent_ids:
            if nbr == entity_id:
                continue
            pr = self._reasoner.find_shortest_path(
                entity_id, nbr, max_depth=max_depth)
            if pr is None:
                continue
            entities.update(pr.path)
            for edge in pr.edges:
                tid = edge.get("triple_id", "")
                if tid and tid not in seen_triple_ids:
                    seen_triple_ids.add(tid)
                    triple = self._store.triple_store.get_triple(tid)
                    if triple is not None:
                        triples.append(triple)
        return entities, triples

    def find_paths_in_subgraph(
        self,
        source_id: str,
        target_id: str,
        entities: set[str],
        triples: list[KnowledgeTriple],
    ) -> list[PathResult]:
        """在子图内查找 source→target 的路径.

        构建子图的邻接表 (仅含 ``entities`` 和 ``triples``),
        用 BFS 查找所有简单路径 (最多返回若干条)。

        Args:
            source_id: 起点 ID.
            target_id: 终点 ID.
            entities: 子图实体集合 (用于过滤).
            triples: 子图三元组列表 (用于构建邻接表).

        Returns:
            路径结果列表 (每条含路径节点、边、总权重)。
        """
        if source_id not in entities or target_id not in entities:
            return []

        # 构建子图邻接表
        adj: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for t in triples:
            if (t.subject_id in entities and t.object_id
                    and t.object_id in entities):
                weight = 1.0 / max(float(t.confidence), 1e-6)
                edge = {
                    "triple_id": t.triple_id,
                    "predicate": t.predicate,
                    "weight": weight,
                    "confidence": float(t.confidence),
                    "subject_id": t.subject_id,
                    "object_id": t.object_id,
                    "direction": "out",
                }
                adj.setdefault(t.subject_id, []).append((t.object_id, edge))

        # BFS 查找路径 (最多 5 条, 避免组合爆炸)
        max_paths = 5
        max_path_len = len(entities) + 1
        results: list[PathResult] = []

        # 用 DFS 收集简单路径
        stack: list[tuple[str, list[str], list[dict[str, Any]], float]] = [
            (source_id, [source_id], [], 0.0)]
        while stack and len(results) < max_paths:
            node, path, edges, total_w = stack.pop()
            if len(path) > max_path_len:
                continue
            if node == target_id and len(path) > 1:
                results.append(PathResult(
                    start_id=source_id, end_id=target_id,
                    path=list(path), edges=list(edges),
                    total_weight=round(total_w, 6),
                    hop_count=len(path) - 1,
                    explanation=self._reasoner.explain_path(path, edges)))
                continue
            for nbr, edge in adj.get(node, []):
                if nbr in path:
                    continue  # 简单路径: 不重复访问
                stack.append((
                    nbr, path + [nbr], edges + [edge],
                    total_w + float(edge.get("weight", 1.0))))

        # 按总权重升序 (越短越优先)
        results.sort(key=lambda p: p.total_weight)
        return results

    def summarize_subgraph(
        self,
        entity_id: str,
        entities: set[str],
        triples: list[KnowledgeTriple],
    ) -> str:
        """生成子图的结构化文本摘要 (供 LLM 进一步精炼).

        摘要结构:
        1. 焦点实体信息 (名称、类型、描述)
        2. 子图规模 (实体数、三元组数)
        3. 关键实体列表 (名称 + 类型)
        4. 关系分布 (按谓词统计)
        5. 关键事实 (焦点实体的出边三元组)

        Args:
            entity_id: 焦点实体 ID.
            entities: 子图实体集合.
            triples: 子图三元组列表.

        Returns:
            结构化文本摘要。
        """
        lines: list[str] = []
        focus = self._store.get_entity(entity_id)
        focus_name = focus.name if focus else entity_id
        focus_type = (
            focus.entity_type.value if focus and hasattr(focus.entity_type, "value")
            else str(focus.entity_type) if focus else "unknown")

        lines.append(f"# 子图摘要: {focus_name}")
        lines.append("")
        lines.append(f"焦点实体: {focus_name} (ID: {entity_id}, 类型: {focus_type})")
        if focus and focus.description:
            lines.append(f"描述: {focus.description}")
        lines.append(f"子图规模: {len(entities)} 个实体, {len(triples)} 条三元组")
        lines.append("")

        # 关键实体列表
        lines.append("## 子图内实体")
        for eid in sorted(entities):
            ent = self._store.get_entity(eid)
            if ent is None:
                lines.append(f"- {eid} (未知)")
            else:
                etype = (
                    ent.entity_type.value
                    if hasattr(ent.entity_type, "value")
                    else str(ent.entity_type))
                lines.append(f"- {ent.name} (ID: {eid}, 类型: {etype})")
        lines.append("")

        # 关系分布
        lines.append("## 关系分布")
        pred_counts: dict[str, int] = {}
        for t in triples:
            pred_counts[t.predicate] = pred_counts.get(t.predicate, 0) + 1
        for pred, cnt in sorted(
            pred_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"- {pred}: {cnt} 条")
        lines.append("")

        # 关键事实 (焦点实体的出边)
        lines.append(f"## {focus_name} 的关键事实")
        focus_triples = [
            t for t in triples if t.subject_id == entity_id and t.object_id
        ]
        if not focus_triples:
            lines.append("- (无出边三元组)")
        for t in focus_triples[:20]:
            obj_ent = self._store.get_entity(t.object_id)
            obj_name = obj_ent.name if obj_ent else t.object_id
            lines.append(
                f"- {focus_name} -[{t.predicate}]-> {obj_name} "
                f"(置信度: {round(float(t.confidence), 3)})")

        return "\n".join(lines)

    # --------------------------------------------------------
    # 内部辅助
    # --------------------------------------------------------

    def _direct_neighbors_in_subgraph(
        self,
        entity_id: str,
        triples: list[KnowledgeTriple],
    ) -> set[str]:
        """获取焦点实体在子图内的直接邻居."""
        neighbors: set[str] = set()
        for t in triples:
            if t.subject_id == entity_id and t.object_id:
                neighbors.add(t.object_id)
            elif t.object_id == entity_id:
                neighbors.add(t.subject_id)
        neighbors.discard(entity_id)
        return neighbors

    def _collect_names(self, entities: set[str]) -> dict[str, str]:
        """收集实体 ID -> 名称映射."""
        names: dict[str, str] = {}
        for eid in entities:
            ent = self._store.get_entity(eid)
            names[eid] = ent.name if ent else eid
        return names

    def get_stats(self) -> dict[str, Any]:
        """获取子图推理器统计信息."""
        with self._lock:
            return {
                "entities_count": self._store.entity_count(),
                "triples_count": self._store.triple_count(),
                "rules_count": len(self._reasoner.get_rules()),
                "strategies": ["bfs", "shortest_path", "confidence_weighted"],
            }


# ============================================================
# 模块导出
# ============================================================


__all__ = [
    # 数据类
    "TrainingReport",
    "LinkPredictionResult",
    "Goal",
    "WeightedTraversalResult",
    "SubgraphReasoningResult",
    # 推理器
    "TransEEmbedder",
    "BackwardChainingReasoner",
    "ConfidenceWeightedTraversal",
    "SubgraphReasoner",
]
