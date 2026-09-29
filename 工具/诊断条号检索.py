"""诊断"《某法规》第X条"这类查询：为什么正确的条款排不到前面。

这类查询看起来是关键词检索的主场——查询里有法规名、有条款号，都是字面信息，
不需要语义理解。所以直觉上 BM25 应该稳赢。

实测不是。这个脚本要回答的就是"为什么"。

做法是**把 BM25 的分数按查询词逐项拆开**。

BM25 的总分是"每个查询词各自贡献"的加和：

    score(文档, 查询) = Σ_t  score_t(文档)

其中 score_t 只取决于词 t 在文档里的词频、词 t 的逆文档频率（IDF）、
以及文档长度归一化。所以只要对每个查询词单独调一次打分，
就能得到一张"谁贡献了多少分"的表——
总分差在哪、被哪个词拉开或拉平，一眼可见。

用法：
    python 工具/诊断条号检索.py
    python 工具/诊断条号检索.py "证券期货投资者适当性管理办法第二十九条"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
# Windows 控制台默认按 GBK 解码，而这个脚本会打印 ✅/⚠️ 这类符号——
# 不加保护的话，它会**直接崩在打印那一步**，而崩溃点常常在干完活之后
# （评测跑完了、钱花完了，明细一条都没落盘）。详见 工具/修控制台编码.py。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

logging.getLogger('pypdf').setLevel(logging.ERROR)

from app.rag.bm25_index import get_bm25_index, tokenize  # noqa: E402

DEFAULT_QUERIES = [
    '证券期货投资者适当性管理办法第二十九条',
    '证券期货投资者适当性管理办法第二条',
    '证券公司监督管理条例第五十七条',
    '基金募集机构投资者适当性管理实施指引第十条',
]


def article_of(text: str) -> str:
    """取切片开头的条号，没有就返回"（不是条文）"。"""

    head = (text or '').strip()
    for end in range(1, min(10, len(head)) + 1):
        if head[:end].endswith('条'):
            candidate = head[:end]
            if candidate.startswith('第'):
                return candidate
            return '（不是条文）'
    return '（不是条文）'


def trace_pipeline(query: str) -> None:
    """把答案在"向量路 → 关键词路 → RRF → 重排"四段里的名次并列出来。

    这是这个脚本最要紧的一段。只看最终结果，所有失败长得都一样；
    把每一段的名次摆在一起，才能看出**是哪一段把正确答案压下去的**。
    """

    from app.config import get_settings
    from app.core.embeddings import embed_query
    from app.core.vector_store import get_vector_store
    from app.rag.bm25_index import get_bm25_index as _index
    from app.rag.reranker import rerank
    from app.rag.retriever import reciprocal_rank_fusion

    settings = get_settings()
    candidate_k = settings.rerank_candidate_k

    vector_hits = get_vector_store().search(
        vector=embed_query(query), top_k=candidate_k
    )
    bm25_hits = _index().search(query, top_k=candidate_k)
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, limit=candidate_k)
    # 重排取 10 条而不是 5 条：为的是能看到"被挤出去的那些"到底打了多少分。
    # 只看前 5 名的话，第 6 名之后发生了什么完全不可见。
    final = rerank(query, fused, top_k=10)

    def locate(hits: list[dict], key: str) -> str:
        for rank, hit in enumerate(hits, start=1):
            if str(hit.get('chunk_id')) == key:
                return str(rank)
        return '未进入'

    correct_id = next(
        (
            str(hit.get('chunk_id'))
            for hit in bm25_hits
            if article_of(hit.get('text')) == '第二十九条'
            and '证券期货投资者适当性管理办法' in str(hit.get('filename'))
        ),
        None,
    )
    if correct_id is None:
        print('没在向量/关键词候选里找到正确切片，跳过链路追踪。')
        return

    rows = [('正确答案（第二十九条）', correct_id)]
    if final:
        rows.append(('最终第 1 名', str(final[0].get('chunk_id'))))

    print('=' * 92)
    print('链路追踪：每一段把候选排在第几名')
    print()
    print(f'{"":<22} {"向量路":>8} {"关键词路":>8} {"RRF 融合":>10} {"重排后":>8}')
    print('-' * 92)
    for label, key in rows:
        fused_rank = locate(fused, key)
        score = next(
            (hit.get('fused_score') for hit in fused if str(hit.get('chunk_id')) == key), 0.0
        )
        print(
            f'{label:<22} {locate(vector_hits, key):>8} {locate(bm25_hits, key):>8} '
            f'{fused_rank + f" ({score:.5f})":>10} {locate(final, key):>8}'
        )

    print()
    print('RRF 的计分方式：只按名次，1/(60+名次)。')
    print('——它看不见"关键词路第 1 名拿了 53.1 分、第 2 名只有 44.9 分"这个差距。')

    print()
    print('重排打分（越高越相关）：')
    for rank, hit in enumerate(final, start=1):
        marker = ' ← 正确答案' if str(hit.get('chunk_id')) == correct_id else ''
        text = ' '.join(str(hit.get('text') or '').split())[:34]
        print(
            f'  {rank:>2}. {float(hit.get("score") or 0):>7.4f}  '
            f'{article_of(hit.get("text")):<10} {text}{marker}'
        )


def analyse(query: str) -> None:
    index = get_bm25_index()
    index.ensure_ready()

    bm25 = index._bm25
    records = index._records
    if bm25 is None or not records:
        print('BM25 索引是空的。')
        return

    tokens = tokenize(query)
    print('=' * 92)
    print(f'查询：{query}')
    print(f'分词（{len(tokens)} 个）：{tokens}')
    print()

    scores = bm25.get_scores(tokens)
    order = sorted(range(len(records)), key=lambda i: -scores[i])[:8]

    print(f'{"名次":<4} {"总分":>8}  {"条号":<12} {"来源文件":<34} 内容开头')
    print('-' * 92)
    for rank, i in enumerate(order, start=1):
        record = records[i]
        print(
            f'{rank:<4} {scores[i]:>8.3f}  {article_of(record["text"]):<12} '
            f'{str(record.get("filename"))[:32]:<34} '
            f'{" ".join(str(record["text"]).split())[:26]}'
        )

    # ---- 正确答案排第几 ----
    #
    # 正确答案 = 来源文件名匹配查询里的法规名，且切片开头就是查询里的那个条号。
    asked = [t for t in tokens if t.startswith('第') and t.endswith('条')]
    asked_article = asked[-1] if asked else None
    ranked = sorted(range(len(records)), key=lambda i: -scores[i])
    correct = [
        i
        for i in ranked
        if (asked_article and article_of(records[i]['text']) == asked_article)
    ]
    if not correct:
        print()
        print(f'⚠️ 索引里没有任何切片以「{asked_article}」开头。')
        return

    target = correct[0]
    print()
    print(
        f'以「{asked_article}」开头的切片共 {len(correct)} 个，'
        f'最高名次是第 {ranked.index(target) + 1} 名：'
        f'{records[target].get("filename")}'
    )

    # ---- 逐词拆解：正确答案 vs 第 1 名 ----
    top = order[0]
    print()
    print('分数构成（每个查询词单独打分，看它给这两片各贡献多少）：')
    print()
    print(f'{"查询词":<16} {"IDF":>7} {"命中片数":>8}  {"→ 第1名":>10} {"→ 正确片":>10}')
    print('-' * 92)

    idf = bm25.idf
    per_token_totals: list[tuple[str, float, float]] = []
    for token in dict.fromkeys(tokens):  # 去重，保持顺序
        contributions = bm25.get_scores([token])
        to_top = float(contributions[top])
        to_target = float(contributions[target])
        hits = sum(1 for value in contributions if value > 0)
        per_token_totals.append((token, to_top, to_target))
        print(
            f'{token:<16} {idf.get(token, 0):>7.2f} {hits:>8}  '
            f'{to_top:>10.3f} {to_target:>10.3f}'
        )

    print('-' * 92)
    top_sum = sum(item[1] for item in per_token_totals)
    target_sum = sum(item[2] for item in per_token_totals)
    print(f'{"合计":<16} {"":>7} {"":>8}  {top_sum:>10.3f} {target_sum:>10.3f}')

    trace_pipeline(query)


def main() -> int:
    parser = argparse.ArgumentParser(description='诊断条号类查询的 BM25 排序')
    parser.add_argument('query', nargs='?', default=None)
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    for query in queries:
        analyse(query)
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
