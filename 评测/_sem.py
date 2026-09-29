"""一次性：看语义匹配在 D7 上的相似度分布，用来定阈值。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '评测'))

from sqlalchemy import select  # noqa: E402

from app.core.embeddings import embed_texts  # noqa: E402
from app.core.postgres import get_session_factory  # noqa: E402
from app.models.wiki_entry import WikiEntry  # noqa: E402
from app.rag.wiki_route import _cosine, _entry_vectors_cached  # noqa: E402
from 评测集 import load_rows, pick  # noqa: E402

session_factory = get_session_factory()
with session_factory() as session:
    entries = list(
        session.execute(select(WikiEntry).where(WikiEntry.status == 'reviewed')).scalars()
    )
    texts = tuple(
        f'{entry.title} {" ".join(entry.triggers or [])}' for entry in entries
    )
    vectors = _entry_vectors_cached(texts)

    rows = load_rows()
    d7 = [row for row in rows if pick(row, 'dimension') == '跨业务线差异']
    others = [row for row in rows if pick(row, 'dimension') in ('条款精确定位', '概念与定义')][:10]

    for label, group in (('D7 对比题', d7), ('其它维度（不该路由）', others)):
        print(f'===== {label} =====')
        for row, query_vector in zip(
            group, embed_texts([pick(r, 'question') for r in group])
        ):
            scored = sorted(
                zip(entries, vectors), key=lambda pair: -_cosine(query_vector, pair[1])
            )
            top_entry, top_vector = scored[0]
            top_score = _cosine(query_vector, top_vector)
            print(
                f'  {top_score:.3f}  {top_entry.slug[:24]:<26} '
                f'{pick(row, "code")} {pick(row, "question")[:24]}'
            )
