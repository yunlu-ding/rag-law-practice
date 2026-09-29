from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.embeddings import embed_texts
from app.core.vector_store import get_vector_store
from app.models.chunk import Chunk
from app.models.document import Document

logger = logging.getLogger(__name__)

"""向量化与索引。

这一层负责把"切片"变成"可检索的向量"，是 RAG 里最花钱的一步——
所以它的设计目标很明确：**尽量少重复花钱**。

        三个具体体现：

        1. **先删旧向量再写新的**。重新索引是"重来一遍"，不是"追加"。
           这一步失败会直接中止整个索引——理由见 vector_store.delete_by_document
           的说明：少了这一步，库里会静默多出一套重复向量。
2. **分批提交，每批提交后回写 vector_id**。
   万一中途失败，已经写好的批次是有记录的，不用从头再来。
3. **失败时保留切片**。向量化失败不代表解析和切分白做了——
   重试只需要重跑向量化（走 /reindex），不用重新上传。
"""

# 每批处理多少个切片。
# 这个值同时决定了两次事情：单次 embedding 请求的大小，和单个数据库事务的大小。
# 取小一点的好处是失败代价低、进度推进细；代价是请求次数多。
INDEX_BATCH_SIZE = 16


class IndexService:
    """向量化与索引的读写。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def index_document(
        self,
        document_id: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """把一份文档的所有切片向量化并写入向量库。

        返回一份统计，包含处理的切片数——它是"这一步到底做了什么"的证据。
        """

        document = self.db.execute(
            select(Document).where(Document.id == document_id)
        ).scalar_one_or_none()
        if document is None:
            raise ValueError(f'文档不存在：{document_id}')

        chunks = list(
            self.db.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id, Chunk.enabled.is_(True))
                .order_by(Chunk.chunk_index)
            ).scalars().all()
        )
        if not chunks:
            logger.warning('[INDEX] 没有可索引的切片: document_id=%s', document_id)
            return {'indexed': 0, 'total': 0}

        store = get_vector_store()
        store.ensure_collection()

        # 文档级元数据。整份文档共用一份，逐片传没有意义（见 VectorStore.add 的说明）。
        document_meta = {
            'legal_level': document.legal_level,
            'level_rank': document.level_rank,
            'validity': document.validity,
            'scope': document.scope,
        }

        # 先清掉这份文档的旧向量，避免新旧两套并存
        store.delete_by_document(document_id)

        total = len(chunks)
        processed = 0

        for start in range(0, total, INDEX_BATCH_SIZE):
            batch = chunks[start : start + INDEX_BATCH_SIZE]
            vectors = embed_texts([chunk.content for chunk in batch])

            payload = [
                {
                    'chunk_id': chunk.id,
                    'document_id': document_id,
                    'filename': document.filename,
                    'page_number': chunk.page_number,
                    'section_title': chunk.section_title,
                    'splitter_name': chunk.splitter_name,
                    'content_type': chunk.content_type,
                    'text': chunk.content,
                }
                for chunk in batch
            ]
            vector_ids = store.add(
                chunks=payload,
                vectors=vectors,
                document_meta=document_meta,
            )

            # 回写向量主键。没有这一步，将来删除文档就找不到它的向量，
            # 只能整库重建——那正是"孤儿向量"的来源。
            for chunk, vector_id in zip(batch, vector_ids, strict=False):
                chunk.vector_id = str(vector_id)
            self.db.commit()

            processed += len(batch)
            if progress_callback is not None:
                progress_callback(processed, total)

        logger.info(
            '[INDEX] 索引完成: document_id=%s filename=%s 切片=%s',
            document_id,
            document.filename,
            processed,
        )
        return {'indexed': processed, 'total': total}

    def delete_document_vectors(self, document_id: str) -> None:
        """删除一份文档的全部向量。"""

        get_vector_store().delete_by_document(document_id)


def vector_search(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """最小可用的向量检索：把问题向量化，再去找最像的切片。

    这一版**只做向量一路**，是刻意的：先把链路打通、能验证，
    再加关键词一路和重排。一次只动一个变量，
    出问题时才能确定是哪里出的问题。
    """

    from app.core.embeddings import embed_query

    vector = embed_query(query)
    return get_vector_store().search(vector=vector, top_k=top_k)
