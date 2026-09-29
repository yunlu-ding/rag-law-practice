from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DocumentItem(BaseModel):
    """文档记录（接口返回用）。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    knowledge_base: str
    filename: str
    file_type: str
    file_size: int | None = None
    file_hash: str | None = None
    status: str = Field(description='处理状态：queued/parsing/parsed/failed 等')
    progress: int = Field(description='整条流水线的进度百分比')
    page_count: int | None = None
    char_count: int | None = None
    chunk_count: int = 0
    summary: str | None = Field(default=None, description='状态说明；失败时是失败原因')
    parse_report: dict[str, Any] | None = Field(default=None, description='解析质检报告')
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hash_preview(self) -> str:
        """指纹前 8 位。

        前端只需要用它来判断"这是不是同一个文件"，
        没必要把 64 位全铺在界面上。
        """

        return (self.file_hash or '')[:8]


class DocumentUploadResponse(BaseModel):
    document: DocumentItem
    message: str = Field(description='给用户看的结果说明')
    duplicated: bool = Field(default=False, description='是否为重复文件被拦截')


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentItem]
