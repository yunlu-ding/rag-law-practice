from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.embeddings import embed_query
from app.core.vector_store import get_vector_store
from app.rag.bm25_index import get_bm25_index
from app.rag.reranker import rerank

logger = logging.getLogger(__name__)

"""混合检索。

流程：

    问题 ─┬─ 向量路：embedding → 余弦相似 → Top-N
          └─ 关键词路：BM25 → Top-N
                   ↓
              RRF 融合（按名次融合）
                   ↓
              取前 candidate_k 条
                   ↓
              重排（可开关，失败退回原顺序）
                   ↓
              Top-K

**为什么用 RRF（倒数排名融合），而不是把两路分数加权相加：**
向量相似度和 BM25 分数的量纲完全不同，加权就得先归一化，
而归一化方式本身又是一个需要调参的主观选择。
RRF 只用名次：1/(k + rank)。它天然免疫量纲问题，也没有需要标定的权重。
代价是丢掉了分数的绝对信息——在这个阶段，稳健比精细重要。

**每一步的耗时都单独记录**，因为"慢"和"不准"是两个问题，
必须能分开看。检索调试台会把它们显示出来。
"""

# RRF 里的平滑常数。取 60 是文献里的常用值：
# 它让"第 1 名和第 2 名"的差距不至于过大，避免单路结果压倒性主导。
RRF_K = 60


@dataclass
class RetrievalOutcome:
    """一次检索的完整结果，含过程数据。

    过程数据和结果一样重要：出问题时你要能回答
    "是向量路没召回到，还是关键词路没召回到，还是重排把它压下去了"。
    只返回最终结果的话，这三个问题的答案就都没有了。
    """

    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    top_k: int = 5
    candidate_k: int = 0
    rerank_enabled: bool = False
    vector_hit_count: int = 0
    bm25_hit_count: int = 0
    fused_count: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    log_id: str | None = None


def retrieve(query: str, *, top_k: int = 5) -> RetrievalOutcome:
    """执行一次混合检索。"""

    settings = get_settings()
    cleaned = (query or '').strip()
    outcome = RetrievalOutcome(
        query=cleaned,
        top_k=top_k,
        rerank_enabled=bool(settings.rerank_enabled),
    )
    if not cleaned:
        return outcome

    # 开了重排就粗筛一个更大的池子，交给重排去精选；
    # 没开重排就按 top_k 的几倍召回即可，池子开大没有意义。
    candidate_k = (
        max(settings.rerank_candidate_k, top_k)
        if outcome.rerank_enabled
        else max(top_k * 4, 10)
    )
    outcome.candidate_k = candidate_k

    # ---- 向量路 ----
    started = time.perf_counter()
    try:
        vector = embed_query(cleaned)
        vector_hits = get_vector_store().search(vector=vector, top_k=candidate_k)
    except Exception as exc:  # noqa: BLE001
        # 单路失败不中断整条检索：另一路可能仍然能给出可用结果。
        # 但要把原因记下来——静默降级是上一个项目里最难查的一类问题。
        logger.exception('[RETRIEVE] 向量路失败: query=%r', cleaned)
        vector_hits = []
        outcome.error = f'向量路失败：{type(exc).__name__}: {exc}'
    outcome.timings_ms['vector'] = int((time.perf_counter() - started) * 1000)
    outcome.vector_hit_count = len(vector_hits)

    # ---- 关键词路 ----
    started = time.perf_counter()
    try:
        bm25_hits = get_bm25_index().search(cleaned, top_k=candidate_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception('[RETRIEVE] 关键词路失败: query=%r', cleaned)
        bm25_hits = []
        outcome.error = (outcome.error or '') + f' 关键词路失败：{type(exc).__name__}: {exc}'
    outcome.timings_ms['bm25'] = int((time.perf_counter() - started) * 1000)
    outcome.bm25_hit_count = len(bm25_hits)

    # ---- 融合 ----
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, limit=candidate_k)
    outcome.fused_count = len(fused)

    # ---- 重排 ----
    started = time.perf_counter()
    final_hits = rerank(cleaned, fused, top_k=top_k)
    outcome.timings_ms['rerank'] = int((time.perf_counter() - started) * 1000)

    outcome.hits = [_with_effect_labels(hit) for hit in final_hits]
    logger.info(
        '[RETRIEVE] 完成: query=%r 向量=%s 关键词=%s 融合=%s 返回=%s 耗时=%s',
        cleaned,
        outcome.vector_hit_count,
        outcome.bm25_hit_count,
        outcome.fused_count,
        len(final_hits),
        outcome.timings_ms,
    )
    return outcome


def _with_effect_labels(hit: dict[str, Any]) -> dict[str, Any]:
    """给检索结果补上"效力层级 / 效力状态"的中文标签。

    为什么要在这里补，而不是让提示词模块自己翻译：

    提示词里有条硬规则——回答"是否违规"时**必须说明依据出自哪一层效力**。
    而模型拿到的只有 `legal_level="department_rule"` 这样的机器值。
    如果让它自己去理解这个值，它只能猜；不给它这个值，它就只能编。

    所以：**在把片段交给模型之前，把机器值翻成人话。**
    这是"数据在哪一层就处理好哪一层"的做法——检索层知道层级，
    就别让提示词层去反推。
    """

    from app.rag.metadata import VALIDITY_LABELS, level_label

    enriched = dict(hit)
    enriched['legal_level_label'] = level_label(hit.get('legal_level'))
    enriched['validity_label'] = VALIDITY_LABELS.get(
        str(hit.get('validity') or 'effective'), '效力未知'
    )
    return enriched


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """按名次融合两路结果。"""

    merged: dict[str, dict[str, Any]] = {}

    for source, hits in (('vector', vector_hits), ('bm25', bm25_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = str(hit.get('chunk_id') or f'{source}:{rank}')
            entry = merged.setdefault(
                key,
                {
                    **hit,
                    'fused_score': 0.0,
                    'retrieval_sources': [],
                    'rank_vector': None,
                    'rank_bm25': None,
                    'vector_score': None,
                    'bm25_score': None,
                },
            )
            entry['fused_score'] += 1.0 / (RRF_K + rank)
            if source not in entry['retrieval_sources']:
                entry['retrieval_sources'].append(source)

            if source == 'vector':
                entry['rank_vector'] = rank
                entry['vector_score'] = hit.get('score')
            else:
                entry['rank_bm25'] = rank
                entry['bm25_score'] = hit.get('score')

            # 补齐两路各自的字段（比如向量路没有 bm25_score）
            for field_name, value in hit.items():
                if field_name in {'retrieval_sources', 'retrieval_source'}:
                    continue
                if entry.get(field_name) in (None, '', []):
                    entry[field_name] = value

    ranked = sorted(merged.values(), key=lambda item: item['fused_score'], reverse=True)
    for position, entry in enumerate(ranked, start=1):
        entry['rank_fused'] = position
        if len(entry['retrieval_sources']) > 1:
            entry['retrieval_source'] = 'hybrid'
            entry['score'] = entry['fused_score']
        elif entry['retrieval_sources'] == ['bm25']:
            entry['retrieval_source'] = 'bm25'
            entry['score'] = entry['bm25_score']
        else:
            entry['retrieval_source'] = 'vector'
            entry['score'] = entry['vector_score']

    return ranked[:limit]
