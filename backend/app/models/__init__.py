"""数据模型。

这里 import 一次所有模型，是为了让 SQLAlchemy 的 metadata 能收集到全部表定义——
建表（create_all）只认已经被 import 过的模型。
新增模型时记得在这里补一行，否则表不会被创建。
"""

from app.models.base import Base, TimestampMixin
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.qa_log import QaLog
from app.models.retrieval_log import RetrievalLog
from app.models.usage import UsageDaily
from app.models.wiki_entry import WikiEntry

__all__ = [
    'Base',
    'TimestampMixin',
    'Document',
    'Chunk',
    'RetrievalLog',
    'QaLog',
    'UsageDaily',
    'WikiEntry',
]
