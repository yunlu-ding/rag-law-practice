from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_database
from app.schemas.chunk import ChunkItem, ChunkListResponse
from app.services.chunk_service import ChunkService

router = APIRouter(prefix='/chunks', tags=['chunks'])


class ChunkUpdateRequest(BaseModel):
    enabled: bool


@router.get('', response_model=ChunkListResponse)
def list_chunks(
    document_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_database),
) -> ChunkListResponse:
    """切片列表，可按文档过滤。"""

    service = ChunkService(db)
    chunks = service.list_chunks(document_id=document_id, limit=limit, offset=offset)
    return ChunkListResponse(
        total=service.count_chunks(document_id=document_id),
        items=[ChunkItem.model_validate(chunk) for chunk in chunks],
    )


@router.get('/{chunk_id}', response_model=ChunkItem)
def get_chunk(chunk_id: str, db: Session = Depends(get_database)) -> ChunkItem:
    service = ChunkService(db)
    chunk = service.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='切片不存在')
    return ChunkItem.model_validate(chunk)


@router.patch('/{chunk_id}', response_model=ChunkItem)
def update_chunk(
    chunk_id: str,
    request: ChunkUpdateRequest,
    db: Session = Depends(get_database),
) -> ChunkItem:
    """启用 / 停用某条切片。

    这是"人工兜底"的入口：自动切分不可能永远正确，
    当某条切片明显切坏了（比如半句话、页眉混进来），
    用户可以把它排除出检索，而不必重建整份文档。
    """

    service = ChunkService(db)
    chunk = service.set_enabled(chunk_id, enabled=request.enabled)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='切片不存在')
    return ChunkItem.model_validate(chunk)
