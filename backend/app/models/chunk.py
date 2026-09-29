from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base, TimestampMixin):
    """切片表。

    这是整个系统里最重要的一张表——**检索的粒度、引用的精度、答案的依据都从这里来**。

    字段设计上有几个刻意的选择：

    - `splitter_name` 单独存一列，而不是塞进元数据：
      因为"换一种切分策略会怎样"是我们要靠实验回答的问题，
      需要能按策略分组统计。塞进 JSON 里就不好查了。
    - `page_number` / `section_title` 单独存列：
      它们是引用展示的主体（"《证券期货投资者适当性管理办法》第 5 页 · 第三章"），
      每次检索都要用，不该埋在 JSON 里。
    - `enabled`：
      允许人工把切坏的片段排除出检索，而不必重建整份文档。
      这是"人工兜底"的入口——自动切分不可能永远正确。
    - `start_offset` / `end_offset`：
      **相对于所属 section 的位置**，用于定位切片在原文中的位置。
      不是全文偏移量，原因见 splitters/base.py 的说明。
    """

    __tablename__ = 'chunk'

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment='切片主键',
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey('document.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='所属文档',
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment='在文档中的顺序编号')
    content: Mapped[str] = mapped_column(Text, nullable=False, comment='切片正文')
    content_type: Mapped[str] = mapped_column(
        String(32), default='text', nullable=False, comment='内容类型：text / table（表格待接入）'
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False, comment='预估 token 数')

    splitter_name: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment='实际使用的切分策略'
    )
    section_index: Mapped[int | None] = mapped_column(Integer, nullable=True, comment='所属 section 序号')
    section_title: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment='所属小节标题；引用展示要用'
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, comment='来源页码')
    article_number: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        index=True,
        comment='所属条款号（如"第二十九条"）。'
        '它是法规场景的定位键：用户问"第X条"时靠它做精确匹配，'
        '而不是靠相似度去猜。无条号的切片（前言、非法规文本）留空',
    )
    start_offset: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment='在所属 section 内的起始位置'
    )
    end_offset: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment='在所属 section 内的结束位置'
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False, comment='扩展元数据'
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment='向量库中的对应主键（阶段四写入）'
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, comment='是否参与检索；可人工排除切坏的片段'
    )

    document: Mapped['Document'] = relationship('Document', backref='chunks')
