"""查询解析的回归测试。

条款级直查能不能用，前提是"法规名 + 条号"被**正确**地从问题里解析出来。
而这个解析是规则写的——规则最容易在你想不到的地方出错
（比如法规名末尾带个"（试行）"，后缀匹配就整体失效了），
所以它必须有测试，而且测试要用**真实的文档清单**跑，不能用手写的假数据。

用法：
    python 工具/测试查询解析.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.core.postgres import get_session_factory  # noqa: E402
from app.rag.citation import load_document_refs, parse_citation  # noqa: E402

CASES = [
    # (问题, 期望锁定的法规名关键词, 期望条号)
    ('证券期货投资者适当性管理办法第二十九条', '证券期货投资者适当性管理办法', '第二十九条'),
    ('《证券期货投资者适当性管理办法》第二十九条怎么规定的', '证券期货投资者适当性管理办法', '第二十九条'),
    ('适当性管理办法第二十九条', '证券期货投资者适当性管理办法', '第二十九条'),
    ('证券法第八十八条', '证券法', '第八十八条'),
    ('《中华人民共和国证券法》第88条', '证券法', '第88条'),
    ('证券公司监督管理条例第五十七条', '证券公司监督管理条例', '第五十七条'),
    ('基金募集机构投资者适当性管理实施指引第十条', '基金募集机构投资者适当性管理实施指引', '第十条'),
    # 下面几条**不该**触发精确直查
    ('第二十九条怎么规定的', None, '第二十九条'),          # 没说哪部法规
    # 没有条号就谈不上精确直查，此时连法规名都不解析——
    # 解析它没有消费者，白做一遍只是给自己一个"看起来更完整"的错觉。
    ('证券期货投资者适当性管理办法问答里怎么解释的', None, None),
    ('《证券法》第八十八条和第八十九条有什么区别', None, None),  # 多个条号
    ('客户适当性管理怎么做', None, None),                   # 没有条号
]

session_factory = get_session_factory()
with session_factory() as session:
    documents = load_document_refs(session)

print(f'语料里的法规 {len(documents)} 部')
print()

passed = failed = 0
for question, expect_doc, expect_article in CASES:
    result = parse_citation(question, documents)
    title = result.document_title or '（未识别）'
    ok_doc = (expect_doc is None and result.document_id is None) or (
        expect_doc is not None and expect_doc in title
    )
    ok_article = result.article_number == expect_article
    ok = ok_doc and ok_article and (result.usable == (expect_doc is not None and expect_article is not None))

    flag = '✅' if ok else '❌'
    if ok:
        passed += 1
    else:
        failed += 1
    print(f'{flag} {question}')
    print(f'     法规={title}')
    print(f'     条号={result.article_number}  可用={result.usable}  说明={result.reason}')
    if not ok:
        print(f'     期望：法规含"{expect_doc}"，条号={expect_article}')
    print()

print(f'通过 {passed} / {passed + failed}')
