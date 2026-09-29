from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class RetrievalLog(Base, TimestampMixin):
    """检索日志。

    这是"可观测性"从"能看"升级到"能积累"的那一步。

    区别很实际：
    - 只有调试台，你只能一次一次手工输入问题、手工看结果；
    - 有了日志，你可以拿一批问题**离线复盘**，
      比较不同参数下同一条查询的召回差异，也能统计哪些问题长期召不回来。

    `items` 里保存的是每次召回的完整过程数据（两路各自的名次、
    融合名次、重排前后名次、分数），因为它正是回答
    "这条答案为什么是这几条"所需要的全部证据。

    注意：保存的正文做了截断（见 retrieval_service 里的说明），
    日志是给人看检索过程的，不是正文的第二份副本。
    """

    __tablename__ = 'retrieval_log'

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment='日志主键',
    )
    query: Mapped[str] = mapped_column(Text, nullable=False, comment='检索问题')
    top_k: Mapped[int] = mapped_column(Integer, default=5, nullable=False, comment='请求返回条数')
    candidate_k: Mapped[int] = mapped_column(Integer, default=0, nullable=False, comment='粗筛候选池大小')
    rerank_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, comment='本次是否启用了重排'
    )

    vector_hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bm25_hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fused_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    returned_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    timings_ms: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False, comment='各阶段耗时：vector / bm25 / rerank'
    )
    items: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False, comment='召回明细（含两路名次与重排前后名次）'
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment='降级记录：哪一路失败了'
    )
