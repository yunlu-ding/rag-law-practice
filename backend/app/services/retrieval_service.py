from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.retrieval_log import RetrievalLog
from app.rag.retriever import RetrievalOutcome, retrieve

logger = logging.getLogger(__name__)

"""检索服务：把"检索"和"记录检索"绑在一起。

为什么记录要放在服务层而不是接口层：
**任何一次检索都必须被记录**，不管它是从调试台来的、从问答来的，
还是以后从别的地方来的。放在接口层，就等于要求每个调用方都记得写日志——
而这种"靠自觉"的约定，迟早会漏。
"""

# 日志里保存的正文长度上限。
#
# 为什么要截断：日志是给人看"检索过程"的，不是正文的第二份副本。
# 完整正文已经存在 chunk 表里，日志里再存一份，
# 既浪费空间，又会让人分不清哪份是准的。
LOG_TEXT_LIMIT = 400


class RetrievalService:
    """检索与检索日志。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def search(self, *, query: str, top_k: int = 5, persist: bool = True) -> RetrievalOutcome:
        """执行检索，并把过程落库。"""

        # 把 db 传进去，才会启用条款级精确直查（见 retriever._try_citation_lookup）。
        outcome = retrieve(query, top_k=top_k, db=self.db)
        if persist:
            outcome.log_id = self._persist(outcome)
        return outcome

    # ---------- 日志 ----------

    def _persist(self, outcome: RetrievalOutcome) -> str | None:
        try:
            log = RetrievalLog(
                query=outcome.query,
                top_k=outcome.top_k,
                candidate_k=outcome.candidate_k,
                rerank_enabled=outcome.rerank_enabled,
                vector_hit_count=outcome.vector_hit_count,
                bm25_hit_count=outcome.bm25_hit_count,
                fused_count=outcome.fused_count,
                returned_count=len(outcome.hits),
                exact_hit_count=outcome.exact_hit_count,
                timings_ms=outcome.timings_ms,
                items=[_shrink(hit) for hit in outcome.hits],
                citation=outcome.citation,
                error=outcome.error,
            )
            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)
            return log.id
        except Exception:  # noqa: BLE001
            # 日志写失败不该让用户的这次检索也失败——检索本身已经完成了。
            # 但必须留下痕迹，否则"日志怎么少了几条"会变成一个查不出的问题。
            self.db.rollback()
            logger.exception('[RETRIEVAL] 检索日志写入失败（检索结果不受影响）')
            return None

    def list_logs(self, *, limit: int = 50, offset: int = 0) -> list[RetrievalLog]:
        statement = (
            select(RetrievalLog)
            .order_by(RetrievalLog.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.db.execute(statement).scalars().all())

    def count_logs(self) -> int:
        from sqlalchemy import func

        return int(self.db.execute(select(func.count()).select_from(RetrievalLog)).scalar_one())

    def get_log(self, log_id: str) -> RetrievalLog | None:
        return self.db.execute(
            select(RetrievalLog).where(RetrievalLog.id == log_id)
        ).scalar_one_or_none()


def _shrink(hit: dict[str, Any]) -> dict[str, Any]:
    """把一条命中记录压缩到适合落库的大小。

    保留全部**用于判断过程**的字段（名次、分数、来源），
    只截断正文。
    """

    return {
        'chunk_id': hit.get('chunk_id'),
        'document_id': hit.get('document_id'),
        'filename': hit.get('filename'),
        'page_number': hit.get('page_number'),
        'section_title': hit.get('section_title'),
        'splitter_name': hit.get('splitter_name'),
        'retrieval_source': hit.get('retrieval_source'),
        'retrieval_sources': list(hit.get('retrieval_sources') or []),
        'score': hit.get('score'),
        'vector_score': hit.get('vector_score'),
        'bm25_score': hit.get('bm25_score'),
        'fused_score': hit.get('fused_score'),
        'rerank_score': hit.get('rerank_score'),
        # 分数**是哪个量纲**。没有它，日志里的 score 就没法用来标定阈值——
        # 实测过：欠费那天向量路挂掉，同一条查询的 score 从 0.32 变成 47.87，
        # 而它在日志里长得跟正常记录一模一样。
        'score_kind': hit.get('score_kind'),
        'rank_vector': hit.get('rank_vector'),
        'rank_bm25': hit.get('rank_bm25'),
        'rank_fused': hit.get('rank_fused'),
        'rank_before_rerank': hit.get('rank_before_rerank'),
        'rank_after_rerank': hit.get('rank_after_rerank'),
        'text': str(hit.get('text') or '')[:LOG_TEXT_LIMIT],
    }
