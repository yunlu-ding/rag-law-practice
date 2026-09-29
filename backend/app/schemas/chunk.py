from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkItem(BaseModel):
    """一条切片（接口返回用）。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int
    content: str
    content_type: str
    token_count: int
    splitter_name: str | None = None
    section_index: int | None = None
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    enabled: bool
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChunkListResponse(BaseModel):
    total: int
    items: list[ChunkItem]


class ChunkStatsResponse(BaseModel):
    """切分质量指标。

    这几个数字是"换一种切法到底有没有变好"的唯一判据，
    所以单独返回，而不是让前端自己算。
    """

    chunk_count: int = 0
    avg_length: int = 0
    max_length: int = 0
    min_length: int = 0
    mid_word_start_count: int = Field(default=0, description='从单词中间开始的切片数（残句）')
    mid_word_ratio: float = Field(default=0.0, description='残句率')
    splitter_usage: dict[str, int] = Field(default_factory=dict, description='各策略切出的片数')


class RechunkRequest(BaseModel):
    splitter: str = Field(
        default='auto',
        description='切分策略：auto=自动判断 / semantic=结构感知 / unstructured=按长度',
    )


class SplitterOptionItem(BaseModel):
    name: str
    description: str
