from __future__ import annotations

import logging
import statistics
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.deps import get_database
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.qa_log import QaLog
from app.models.retrieval_log import RetrievalLog
from app.models.usage import UsageDaily

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/stats', tags=['stats'])

# 统计只取最近这么多次记录。
#
# 为什么不做全量聚合：**工作台的总览要的是"当前状态"，不是审计报表。**
# 全量聚合在数据涨起来之后会拖慢首屏，而首屏慢是工作台最不该有的毛病。
# 需要看全量的时候，去检索调试台翻历史，那里本来就是干这个的。
RECENT_LIMIT = 300


def _distribution(values: list[int]) -> dict[str, int]:
    """把一组耗时压成分位数。

    为什么用分位数而不是平均值：**平均值会被个别长尾拉高**，
    看的人会以为"系统普遍很慢"，其实只是少数几次冷启动。
    分位数能同时说清"通常多快"和"最差多慢"。
    """

    if not values:
        return {'p50': 0, 'p90': 0, 'max': 0, 'count': 0}
    ordered = sorted(values)

    def percentile(ratio: float) -> int:
        index = min(int(len(ordered) * ratio), len(ordered) - 1)
        return int(ordered[index])

    return {
        'p50': percentile(0.5),
        'p90': percentile(0.9),
        'max': int(max(ordered)),
        'count': len(ordered),
    }


@router.get('')
def get_stats(db: Session = Depends(get_database)) -> dict[str, Any]:
    """工作台总览用的聚合数据。

    一次请求返回全部，而不是让前端拼五六个接口——
    总览页要的是"一眼看完"，多一次往返就多一次闪烁。
    """

    # ---------- 知识库构成 ----------
    documents = list(
        db.execute(select(Document).order_by(Document.chunk_count.desc())).scalars().all()
    )
    chunk_total = int(db.execute(select(func.count()).select_from(Chunk)).scalar_one())
    chunk_enabled = int(
        db.execute(select(func.count()).select_from(Chunk).where(Chunk.enabled.is_(True))).scalar_one()
    )
    avg_length = int(
        db.execute(select(func.avg(func.length(Chunk.content)))).scalar_one() or 0
    )
    by_splitter = {
        (name or '未知'): count
        for name, count in db.execute(
            select(Chunk.splitter_name, func.count()).group_by(Chunk.splitter_name)
        ).all()
    }

    # ---------- 检索 ----------
    retrieval_logs = list(
        db.execute(
            select(RetrievalLog).order_by(RetrievalLog.created_at.desc()).limit(RECENT_LIMIT)
        ).scalars().all()
    )
    retrieval_latencies: list[int] = []
    source_counter: dict[str, int] = {}
    rerank_on = 0
    degraded = 0
    for log in retrieval_logs:
        timings = log.timings_ms or {}
        # retrieval_log 里没有"总耗时"字段，只有各阶段耗时，所以在这里加总。
        # 好处是顺便能看出时间花在哪一段上。
        total = sum(v for v in timings.values() if isinstance(v, int))
        if total:
            retrieval_latencies.append(total)
        if log.rerank_enabled:
            rerank_on += 1
        if log.error:
            degraded += 1
        for item in log.items or []:
            for source in item.get('retrieval_sources') or []:
                source_counter[source] = source_counter.get(source, 0) + 1

    # ---------- 问答 ----------
    qa_logs = list(
        db.execute(select(QaLog).order_by(QaLog.created_at.desc()).limit(RECENT_LIMIT)).scalars().all()
    )
    conclusion_counter: dict[str, int] = {}
    qa_latencies: list[int] = []
    refused = 0
    retrieval_failed = 0
    with_unknown_citation = 0
    with_citation = 0
    for log in qa_logs:
        key = log.conclusion or '（未得到结构化结论）'
        conclusion_counter[key] = conclusion_counter.get(key, 0) + 1
        if log.latency_ms:
            qa_latencies.append(log.latency_ms)
        if log.refused:
            refused += 1
        if log.retrieval_failed:
            retrieval_failed += 1
        if log.citations:
            with_citation += 1
        if log.unknown_citations:
            with_unknown_citation += 1

    qa_total = len(qa_logs)

    # ---------- 用量 ----------
    usage_rows = list(
        db.execute(select(UsageDaily).order_by(UsageDaily.usage_date)).scalars().all()
    )

    return {
        'knowledge_base': {
            'documents': [
                {
                    'filename': doc.filename,
                    'chunks': doc.chunk_count,
                    'status': doc.status,
                    'file_type': doc.file_type,
                    'file_size': doc.file_size,
                }
                for doc in documents
            ],
            'documents_total': len(documents),
            'chunks_total': chunk_total,
            'chunks_enabled': chunk_enabled,
            'chunk_avg_length': avg_length,
            'chunks_by_splitter': by_splitter,
        },
        'retrieval': {
            'total': len(retrieval_logs),
            'recent_limit': RECENT_LIMIT,
            'latency': _distribution(retrieval_latencies),
            # 各阶段耗时单独给中位数，这样"慢在哪一段"能直接看出来
            'stage_median': {
                stage: int(
                    statistics.median(
                        [log.timings_ms.get(stage, 0) for log in retrieval_logs if log.timings_ms]
                        or [0]
                    )
                )
                for stage in ('vector', 'bm25', 'rerank')
            },
            'sources': source_counter,
            'rerank_on': rerank_on,
            'degraded': degraded,
        },
        'qa': {
            'total': qa_total,
            'by_conclusion': conclusion_counter,
            'refused': refused,
            'retrieval_failed': retrieval_failed,
            'with_citation': with_citation,
            'with_unknown_citation': with_unknown_citation,
            'latency': _distribution(qa_latencies),
        },
        'usage': {
            'daily': [
                {
                    'date': row.usage_date.isoformat(),
                    'qa_count': row.qa_count,
                    'retrieval_count': row.retrieval_count,
                    'blocked_count': row.blocked_count,
                }
                for row in usage_rows
            ]
        },
    }
