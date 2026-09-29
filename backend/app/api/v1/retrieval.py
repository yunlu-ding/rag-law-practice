from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.deps import enforce_limits, get_database
from app.core.limits import record_usage
from app.services.index_service import vector_search
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/retrieval', tags=['retrieval'])


class VectorSearchRequest(BaseModel):
    query: str = Field(description='检索问题')
    top_k: int = Field(default=5, ge=1, le=50, description='返回条数')


class RetrievalHit(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    article_number: str | None = Field(default=None, description='所属条款号，如"第二十九条"')
    legal_level: str | None = None
    legal_level_label: str | None = None
    validity: str | None = None
    validity_label: str | None = None
    section_title: str | None = None
    splitter_name: str | None = None
    content_type: str | None = None
    text: str | None = None
    score: float | None = Field(
        default=None,
        description='相关性分数。条款精确命中没有这个分数（它不是相似度问题），为 null',
    )
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    score_kind: str | None = Field(
        default=None,
        description='score 属于哪种分数：exact / rerank / fused。'
                    'score 是混合量纲的展示字段，判定时只能认 rerank',
    )
    rank_bm25: int | None = None
    rank_fused: int | None = None
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    rank_vector: int | None = None
    retrieval_source: str | None = None
    vector_id: int | None = None


class HybridSearchRequest(BaseModel):
    query: str = Field(description='检索问题')
    top_k: int = Field(default=5, ge=1, le=50, description='返回条数')


class HybridSearchResponse(BaseModel):
    query: str
    log_id: str | None = Field(default=None, description='本次检索的日志 ID，可在历史里回看')
    top_k: int
    candidate_k: int = Field(description='粗筛候选池大小')
    rerank_enabled: bool
    vector_hit_count: int
    bm25_hit_count: int
    fused_count: int
    exact_hit_count: int = Field(default=0, description='条款级精确直查命中的切片数')
    citation: dict[str, Any] | None = Field(
        default=None,
        description='查询解析结果：识别到的法规名与条号；解析不出来时为 null',
    )
    timings_ms: dict[str, int] = Field(default_factory=dict, description='各阶段耗时')
    error: str | None = Field(default=None, description='降级记录：哪一路失败了')
    items: list[RetrievalHit]


class RetrievalLogItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    query: str
    top_k: int
    candidate_k: int
    rerank_enabled: bool
    vector_hit_count: int
    bm25_hit_count: int
    fused_count: int
    returned_count: int
    exact_hit_count: int = 0
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    citation: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    items: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalLogListResponse(BaseModel):
    total: int
    items: list[RetrievalLogItem]


class VectorSearchResponse(BaseModel):
    query: str
    hit_count: int
    items: list[RetrievalHit]


@router.post('/vector-search', response_model=VectorSearchResponse)
def vector_search_endpoint(
    request: VectorSearchRequest,
    db: Session = Depends(get_database),
) -> VectorSearchResponse:
    """最小可用的向量检索。

    ⚠️ 这一版**只做向量一路**，是刻意的中间状态：
    先把"问题 → 向量 → 命中切片"这条链路跑通并能验证，
    下一步再加关键词一路、RRF 融合和重排。
    一次只动一个变量，出问题时才能确定是哪里出的问题。

    等混合检索做完，这个接口会被正式的 /search 取代——
    但那时候它仍然是排查问题的好工具：
    **只看向量路的结果，可以判断"关键词路有没有帮上忙"。**
    """

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='检索问题不能为空')

    try:
        hits = vector_search(query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception('[RETRIEVAL] 向量检索失败: query=%r', query)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'向量检索失败：{type(exc).__name__}: {exc}',
        ) from exc

    return VectorSearchResponse(
        query=query,
        hit_count=len(hits),
        items=[RetrievalHit.model_validate(hit) for hit in hits],
    )


@router.post(
    '/search',
    response_model=HybridSearchResponse,
    dependencies=[Depends(enforce_limits('retrieval'))],
)
def hybrid_search(
    request: HybridSearchRequest,
    db: Session = Depends(get_database),
) -> HybridSearchResponse:
    """混合检索：向量 + 关键词 → RRF 融合 → 重排。

    返回里除了结果，还有**过程数据**：两路各自召回多少条、
    融合后多少条、每一步花了多少毫秒、有没有哪一路降级了。

    这些字段不是调试用的附属品，而是这个产品能不能迭代的关键：
    没有它们，"检索效果不好"就永远只能是一句感觉。
    """

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='检索问题不能为空')

    try:
        outcome = RetrievalService(db).search(query=query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception('[RETRIEVAL] 混合检索失败: query=%r', query)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'检索失败：{type(exc).__name__}: {exc}',
        ) from exc

    record_usage(kind='retrieval')

    return HybridSearchResponse(
        query=outcome.query,
        log_id=outcome.log_id,
        top_k=outcome.top_k,
        candidate_k=outcome.candidate_k,
        rerank_enabled=outcome.rerank_enabled,
        vector_hit_count=outcome.vector_hit_count,
        bm25_hit_count=outcome.bm25_hit_count,
        fused_count=outcome.fused_count,
        exact_hit_count=outcome.exact_hit_count,
        citation=outcome.citation,
        timings_ms=outcome.timings_ms,
        error=outcome.error,
        items=[RetrievalHit.model_validate(hit) for hit in outcome.hits],
    )


@router.get('/logs', response_model=RetrievalLogListResponse)
def list_retrieval_logs(
    limit: int = 30,
    offset: int = 0,
    db: Session = Depends(get_database),
) -> RetrievalLogListResponse:
    """检索历史。

    这一页的价值在于：**把"能看"变成"能积累"**。
    调试台只能一次一次手工试；有了历史，才能拿一批问题离线复盘，
    比较不同参数下同一条查询的差异。
    """

    service = RetrievalService(db)
    logs = service.list_logs(limit=limit, offset=offset)
    return RetrievalLogListResponse(
        total=service.count_logs(),
        items=[RetrievalLogItem.model_validate(log) for log in logs],
    )


@router.get('/logs/{log_id}', response_model=RetrievalLogItem)
def get_retrieval_log(log_id: str, db: Session = Depends(get_database)) -> RetrievalLogItem:
    service = RetrievalService(db)
    log = service.get_log(log_id)
    if log is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='检索日志不存在')
    return RetrievalLogItem.model_validate(log)
