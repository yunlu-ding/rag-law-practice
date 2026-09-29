from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.chunker import ChunkRecord
from app.rag.bm25_index import get_bm25_index

logger = logging.getLogger(__name__)


class ChunkService:
    """切片相关的读写。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def list_chunks(
        self,
        *,
        document_id: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[Chunk]:
        statement = select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)
        if document_id:
            statement = statement.where(Chunk.document_id == document_id)
        statement = statement.offset(offset).limit(limit)
        return list(self.db.execute(statement).scalars().all())

    def count_chunks(self, *, document_id: str | None = None) -> int:
        from sqlalchemy import func

        statement = select(func.count()).select_from(Chunk)
        if document_id:
            statement = statement.where(Chunk.document_id == document_id)
        return int(self.db.execute(statement).scalar_one())

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.db.execute(select(Chunk).where(Chunk.id == chunk_id)).scalar_one_or_none()

    def replace_chunks(self, *, document_id: str, records: list[ChunkRecord]) -> int:
        """用新的切片替换这份文档的所有旧切片。

        **先清空再写入**，而不是"追加"：
        重新切分是一个"重来一遍"的操作，如果保留旧切片，
        库里会同时存在两套切法切出来的内容，检索结果会互相污染，
        而且用户完全看不出来为什么同一条内容出现了两次。

        清空和写入在同一个事务里完成——中途失败就整体回滚，
        不会留下"删了一半"的文档。

        ⚠️ 这里必须把**文档级信息**（文件名、标题、层级）写进每个切片的
        metadata_json，原因是 BM25 要用它：

        关键词检索的可信度有一半来自"这条来自哪份文件"。
        合规问答里高频出现的正是"《证券法》第八十八条怎么规定的"这种
        带文件名的问法，而正文里可能一次都没写"证券法"三个字
        （它写的是"本法"）。

        这里踩过一次：bm25_index 的 _lexical_text 一直在读
        metadata_json['filename'] 并给它双倍权重，而写入侧从来没放过这个键。
        结果是权重逻辑形同虚设，而且**不报错、看不出来**——
        只会觉得"按文件名搜怎么搜不到"。
        """

        document = self.db.execute(
            select(Document).where(Document.id == document_id)
        ).scalar_one_or_none()
        document_meta = {
            'filename': document.filename if document else None,
            'title': document.title if document else None,
            'legal_level': document.legal_level if document else None,
            'validity': document.validity if document else None,
        }

        self.db.execute(delete(Chunk).where(Chunk.document_id == document_id))

        chunk_models = [
            Chunk(
                document_id=document_id,
                chunk_index=record.chunk_index,
                content=record.content,
                content_type=record.content_type,
                token_count=record.token_count,
                splitter_name=record.splitter_name,
                section_index=record.section_index,
                section_title=(record.section_title or None),
                page_number=record.page_number,
                start_offset=record.start_offset,
                end_offset=record.end_offset,
                metadata_json={
                    'section_type': record.section_type,
                    'splitter_name': record.splitter_name,
                    **document_meta,
                },
                enabled=True,
            )
            for record in records
        ]
        self.db.add_all(chunk_models)
        self.db.commit()

        # 切片一变，关键词索引就过期了。
        # 不标记的话，检索会命中已经不存在的内容——
        # 表现为"搜到一条，点开是空的"，而且极难查原因。
        get_bm25_index().mark_dirty(f'chunks_replaced:{document_id}')

        logger.info('[CHUNK] 已写入切片: document_id=%s 数量=%s', document_id, len(chunk_models))
        return len(chunk_models)

    def set_enabled(self, chunk_id: str, *, enabled: bool) -> Chunk | None:
        """启用 / 停用某条切片。

        用于人工把切坏的片段排除出检索。自动切分不可能永远正确，
        留一个"人工纠正"的入口，比要求用户重新调参重切一遍现实得多。
        """

        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return None
        chunk.enabled = enabled
        self.db.commit()
        self.db.refresh(chunk)
        get_bm25_index().mark_dirty(f'chunk_enabled_changed:{chunk_id}')
        return chunk
