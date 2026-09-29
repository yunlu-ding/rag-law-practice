"""把混合检索的每一路拆开看，定位"检索不准"到底出在哪一步。

混合检索的链路是：

    向量路 ─┐
            ├─ RRF 融合 ─ 重排 ─ 最终结果
    BM25路 ─┘

最终结果不对时，有四种可能，处理办法完全不同：

    1. 向量路本身没召回到        → embedding 或语料质量问题
    2. BM25 路没召回到           → 分词或字段加权问题
    3. 两路都召回到了但融合掉了   → RRF 参数问题
    4. 融合后进了候选、但被重排压下去 → 重排模型问题

只看最终 Top-K 的话，这四个原因长得一模一样。
这个脚本把每一路的原始名次都打出来，直接指出是哪一种。

用法：
    python 工具/诊断检索.py "证券期货投资者适当性管理办法第二十九条"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
logging.getLogger('pypdf').setLevel(logging.ERROR)

from app.core.embeddings import embed_query  # noqa: E402
from app.core.vector_store import get_vector_store  # noqa: E402
from app.rag.bm25_index import get_bm25_index, tokenize  # noqa: E402


def brief(hit: dict) -> str:
    text = ' '.join(str(hit.get('text') or '').split())
    return f'{str(hit.get("filename"))[:26]:<28} | {text[:58]}'


def main() -> int:
    parser = argparse.ArgumentParser(description='诊断一路检索')
    parser.add_argument('query')
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()

    print(f'问题：{args.query}')
    print()

    # ---- BM25 路 ----
    #
    # 分词结果要单独打出来。中文检索里"召不回来"最常见的原因就是分词把
    # 关键短语切碎了（"适当性" 切成 "适当" + "性"），
    # 而这件事从检索结果里看不出来。
    tokens = tokenize(args.query)
    print(f'BM25 查询分词（{len(tokens)} 个）：{tokens}')
    print()

    bm25_hits = get_bm25_index().search(args.query, top_k=args.top)
    print(f'--- BM25 路 Top{args.top} ---')
    if not bm25_hits:
        print('  （空）')
    for rank, hit in enumerate(bm25_hits, start=1):
        print(f'  {rank}. bm25={hit["score"]:.3f}  {brief(hit)}')
    print()

    # ---- 向量路 ----
    vector = embed_query(args.query)
    vector_hits = get_vector_store().search(vector=vector, top_k=args.top)
    print(f'--- 向量路 Top{args.top} ---')
    if not vector_hits:
        print('  （空）')
    for rank, hit in enumerate(vector_hits, start=1):
        print(f'  {rank}. cos={hit["score"]:.4f}  {brief(hit)}')
    print()

    from app.rag.retriever import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, limit=args.top)
    print(f'--- RRF 融合后 Top{args.top} ---')
    for rank, hit in enumerate(fused[: args.top], start=1):
        print(f'  {rank}. rrf={hit.get("rrf_score", 0):.5f}  {brief(hit)}')

    # ---- 重排前后对比 ----
    #
    # 这一段的用途是回答"重排到底在帮忙还是在添乱"。
    # 只看最终结果看不出来：重排失败会退回原顺序，所以最终顺序**看起来总是合理的**。
    # 必须把重排前的名次和重排后的名次并排放。
    print()
    print('--- 重排前 vs 重排后 ---')
    from app.config import get_settings
    from app.rag.reranker import rerank

    settings = get_settings()
    print(f'（RERANK_ENABLED={settings.rerank_enabled}，模型 {settings.rerank_model}）')

    def key_of(hit: dict) -> str:
        return str(hit.get('chunk_id') or '')

    before = [key_of(hit) for hit in fused]
    reranked = rerank(args.query, fused, top_k=args.top)
    after = [key_of(hit) for hit in reranked]

    for rank, chunk_id in enumerate(after, start=1):
        previous = before.index(chunk_id) + 1 if chunk_id in before else None
        hit = next(item for item in reranked if key_of(item) == chunk_id)
        movement = f'（重排前第 {previous} 名）' if previous else '（重排前不在候选里）'
        print(f'  {rank}. {brief(hit)}  {movement}')

    # ---- 生产链路本身 ----
    #
    # 上面两步用的是"每路只取 top-N"，而线上用的是 candidate_k（默认 100）。
    # 候选池大小会改变重排的结果，所以必须把真实链路也跑一遍，
    # 否则诊断出来的结论和线上看到的会对不上。
    print()
    print('--- retrieve()（线上真实链路，candidate_k=%s）---' % settings.rerank_candidate_k)
    from app.rag.retriever import retrieve

    outcome = retrieve(args.query, top_k=args.top)
    print(f'  向量路 {outcome.vector_hit_count} 条，BM25 路 {outcome.bm25_hit_count} 条，'
          f'融合后 {outcome.candidate_k} 条')
    for rank, hit in enumerate(outcome.hits, start=1):
        sources = '+'.join(hit.get('retrieval_sources') or [])
        print(f'  {rank}. score={hit.get("score", 0):.4f} [{sources}] {brief(hit)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
