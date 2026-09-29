from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import enforce_limits, get_database
from app.core.limits import record_usage
from app.models.qa_log import QaLog
from app.services.qa_service import QaService

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/qa', tags=['qa'])


class AskRequest(BaseModel):
    question: str = Field(description='用户问题')
    top_k: int = Field(default=5, ge=1, le=20, description='喂给模型的证据条数')
    session_id: str | None = Field(
        default=None,
        description=(
            '会话 ID。传了就会带上这个会话的历史，并把追问改写成可独立检索的问题；'
            '不传就是单轮问答'
        ),
    )
    refuse_threshold: float | None = Field(
        default=None,
        description='拒答阈值。不传则用配置里的值。传 0 等于不拒答（任何分数都放行）',
    )


class Citation(BaseModel):
    chunk_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    score: float | None = None
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    session_id: str | None = None
    standalone_query: str | None = Field(
        default=None, description='追问改写后实际用于检索的问题'
    )
    rewritten: bool = Field(default=False, description='本次是否发生了追问改写')
    history_turns: int = Field(default=0, description='带入了多少轮历史')
    refused: bool = Field(description='是否走了拒答分支')
    refusal_reason: str | None = None
    conclusion: str | None = Field(default=None, description='违反 / 不违反 / 无法判断 / 说明')
    clause: str | None = None
    reasoning: str | None = None
    assumption: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    unknown_citations: list[str] = Field(
        default_factory=list,
        description='模型引用了不存在的片段编号 —— 可检测的编造',
    )
    parse_ok: bool = Field(description='模型输出是否解析成功')
    retrieval_failed: bool = Field(
        default=False,
        description='检索链路故障导致无法作答。这是系统故障，不是知识库里没有内容',
    )
    retrieval: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    log_id: str | None = None
    error: str | None = None


class QaLogItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str | None = None
    question: str
    conclusion: str | None = None
    clause: str | None = None
    reasoning: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)
    refused: bool
    refusal_reason: str | None = None
    parse_ok: bool
    retrieval_failed: bool = False
    latency_ms: int | None = None
    created_at: datetime


class QaLogListResponse(BaseModel):
    total: int
    items: list[QaLogItem]


@router.post('/ask', response_model=AskResponse, dependencies=[Depends(enforce_limits('qa'))])
def ask(request: AskRequest, db: Session = Depends(get_database)) -> AskResponse:
    """问答：检索 → 拒答判断 → 生成 → 引用还原。

    返回体里除了答案，还带三样东西，它们都是"这是不是一次可信回答"的证据：

    - `citations`：已还原的引用（文件名 + 页码 + 原文），可核对；
    - `unknown_citations`：模型引用了不存在的片段编号的清单，
      非空就说明这次回答出现过编造引用；
    - `retrieval`：当时的召回情况与各阶段耗时。
    """

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='问题不能为空')

    try:
        outcome = QaService(db).ask(
            question=question,
            top_k=request.top_k,
            refuse_threshold=request.refuse_threshold,
            session_id=request.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception('[QA] 问答失败: question=%r', question)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'问答失败：{type(exc).__name__}: {exc}',
        ) from exc

    # 用量记录放在最后：只有真的花掉了一次调用才计数。
    # 放在最前面会把被拒答、被校验拦下的请求也算进去，用量就虚高了。
    record_usage(kind='qa', completion_tokens=outcome.tokens)

    return AskResponse(
        question=outcome.question,
        session_id=outcome.session_id,
        standalone_query=outcome.standalone_query,
        rewritten=outcome.rewritten,
        history_turns=outcome.history_turns,
        refused=outcome.refused,
        refusal_reason=outcome.refusal_reason,
        conclusion=outcome.conclusion,
        clause=outcome.clause,
        reasoning=outcome.reasoning,
        assumption=outcome.assumption,
        citations=[Citation.model_validate(item) for item in outcome.citations],
        unknown_citations=outcome.unknown_citations,
        parse_ok=outcome.parse_ok,
        retrieval_failed=outcome.retrieval_failed,
        retrieval=outcome.retrieval,
        latency_ms=outcome.latency_ms,
        log_id=outcome.log_id,
        error=outcome.error,
    )


@router.get('/logs', response_model=QaLogListResponse)
def list_qa_logs(
    limit: int = 30,
    offset: int = 0,
    session_id: str | None = None,
    db: Session = Depends(get_database),
) -> QaLogListResponse:
    """问答记录。

    这张表是评测的数据来源：可以按结论类型、是否拒答、
    有没有出现未知引用来筛样本，把人工判卷的力气花在最值得看的地方。
    """

    from sqlalchemy import func

    count_statement = select(func.count()).select_from(QaLog)
    statement = select(QaLog)
    if session_id:
        count_statement = count_statement.where(QaLog.session_id == session_id)
        statement = statement.where(QaLog.session_id == session_id)
    total = int(db.execute(count_statement).scalar_one())
    statement = statement.order_by(QaLog.created_at.desc()).offset(offset).limit(limit)
    logs = list(db.execute(statement).scalars().all())
    return QaLogListResponse(
        total=total,
        items=[QaLogItem.model_validate(log) for log in logs],
    )
