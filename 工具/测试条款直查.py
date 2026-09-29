"""条款级精确直查的回归测试。

重点测的是**边界**，因为这类功能的失败方式都很安静：

  - 阿拉伯数字（"第88条"）能不能匹配中文数字的条号；
  - 引用了**不存在**的条号时，是不是安静退回普通检索，而不是硬凑一个答案；
  - 没说是哪部法规时，是不是不做直查（只按条号过滤会召回十几份文件的"第二十九条"）；
  - 普通语义查询是不是完全不受影响。

用法：
    python 工具/测试条款直查.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.core.postgres import get_session_factory  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

CASES = [
    '证券期货投资者适当性管理办法第二十九条',
    '《中华人民共和国证券法》第88条怎么规定的',
    '证券公司监督管理条例第五十七条',
    '证券期货投资者适当性管理办法第八十条',      # 该法规只有 43 条，不存在
    '境外市场的投资者适当性要求是什么',            # 普通语义查询，不该走直查
    '第二十九条怎么规定的',                        # 没说是哪部法规
]

session_factory = get_session_factory()
with session_factory() as session:
    service = RetrievalService(session)
    for query in CASES:
        outcome = service.search(query=query, top_k=3, persist=False)
        citation = outcome.citation or {}
        print('=' * 88)
        print(f'问：{query}')
        print(f'解析：法规={citation.get("document_title") or "—"}  '
              f'条号={citation.get("article_number") or "—"}  '
              f'直查命中={outcome.exact_hit_count} 片')
        print(f'     说明：{citation.get("reason") or "—"}')
        for rank, hit in enumerate(outcome.hits, start=1):
            sources = '+'.join(hit.get('retrieval_sources') or [])
            label = hit.get('article_number') or '—'
            text = ' '.join(str(hit.get('text') or '').split())[:40]
            print(f'  {rank}. [{sources:<12}] 条款={label:<8} {text}')
        print()
