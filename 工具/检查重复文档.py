"""检查库里有没有同名重复的文档。

为什么需要它：批量入库曾经出过一次同名重复——`--force` 按**内容哈希**找旧记录，
而文件内容一变（换语料、改解析规则、重新生成），哈希对不上，
于是"找不到旧的"→ 新建一份 → 同一个文件在库里出现两份，
内容还各不相同。表现是检索结果里同一份法规出现两遍，且不报错。

入库逻辑已经修成按**文件名**替换了，这个脚本留作事后核对——
**能自己检查自己的数据，比相信自己不会犯错可靠。**
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.document import Document  # noqa: E402

session_factory = get_session_factory()
with session_factory() as session:
    documents = session.execute(select(Document).order_by(Document.filename)).scalars().all()

print(f'文档总数：{len(documents)}')
counts = Counter(document.filename for document in documents)
duplicates = {name: count for name, count in counts.items() if count > 1}
if not duplicates:
    print('没有同名重复。')
else:
    print('同名重复：')
    for name, count in duplicates.items():
        print(f'  {name}：{count} 份')
        for document in documents:
            if document.filename == name:
                print(
                    f'     id={document.id[:8]} 状态={document.status} '
                    f'切片={document.chunk_count} 入库={document.created_at:%Y-%m-%d %H:%M:%S}'
                )
