"""诊断 Wiki 词条路由：问题进来了，路由把它送给了哪条词条，送对没有。

为什么要有这个工具：

语义路由上线之后出现了**最难受的一种现象——指标不动**。
路由命中数从 10 涨到 13，但评测通过率一点没变（81% / 60%）。
"命中更多但结果没变"有两种可能，而它们要采取的行动完全相反：

  A. 路由送对了，但词条内容**没覆盖**评测锚点 → 该去补词条；
  B. 路由**送错了**词条（送去的词条和问题不是一回事）→ 该去改路由。

光看通过率这两种情况长得一模一样，所以必须把中间那一步摊开看：
**每个问题 → 路由选中的 slug + 相似度 → 正确 slug 的相似度 →
正确词条的依据条款里有没有评测锚点。**

用法：
    python 工具/诊断词条路由.py            # 只跑"跨业务线差异"（D7）
    python 工具/诊断词条路由.py --only D7
    python 工具/诊断词条路由.py --all      # 全评测集（看误路由）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '评测'))

# Windows 控制台默认是 GBK，打印 ✅/❌ 会直接抛 UnicodeEncodeError 把脚本打断。
# 在入口处改掉 stdout 编码，比要求每个人记得先敲 chcp 65001 可靠。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from sqlalchemy import select  # noqa: E402

from app.core.embeddings import embed_texts  # noqa: E402
from app.core.postgres import get_session_factory  # noqa: E402
from app.models.wiki_entry import WikiEntry  # noqa: E402
from app.rag.wiki_route import (  # noqa: E402
    KEYWORD_BONUS_PER_CHAR,
    _cosine,
    _entry_vectors_cached,
    _keyword_hit,
    _bare,
    is_comparison,
    match_entry_with_vector,
)
from 评测集 import load_rows, pick  # noqa: E402

SEPARATOR = '｜'

# 人工标注的"这题该走哪条词条"。
# 之所以要人来标：**没有一条判据能从问题文本推出正确词条**——
# "三条业务线在投资者分类上的要求一样吗"该走 investor-classification，
# 这件事只有读过评测集的人知道。让代码去猜，就是把评测本身也变成待验证的东西。
#
# 标注完发现两件事，都值得记：
#   1. 我第一版凭印象标，**标错了 4 条**（D7-13/14/15 其实路由是对的）。
#      所以这份表必须对着问题原文一条条标，不能凭印象；
#   2. 有 1 条（D7-02）**没有对应词条**——冷静期这个主题根本没写过。
#      它在通过率里表现为"没通过"，但根因不是路由也不是检索，是**选题漏了**。
EXPECTED = {
    'D7-01': 'investor-classification-three-lines',
    'D7-02': None,  # 冷静期：没有对应词条（选题漏了）
    'D7-03': 'securities-vs-futures-suitability',
    'D7-04': 'industry-association-ownership',
    'D7-05': 'risk-rating-comparison',
    'D7-06': 'investor-classification-three-lines',
    'D7-07': 'revisit-requirements-comparison',
    'D7-08': 'self-regulation-division',
    'D7-09': 'sales-staff-management-comparison',
    'D7-10': 'record-retention-comparison',
    'D7-11': 'matching-principles-comparison',
    'D7-12': 'penalty-comparison',
    'D7-13': 'industry-association-ownership',
    'D7-14': 'risk-catalogue-responsibility',
    'D7-15': 'lowest-risk-tolerance-investor',
}

ARTICLE = re.compile(r'第([一二三四五六七八九十百零〇\d]+)条')


def normalize(text: str) -> str:
    return re.sub(r'[\s\u3000]+', '', text or '')


def main() -> int:
    parser = argparse.ArgumentParser(description='诊断词条路由')
    parser.add_argument('--only', default='D7')
    parser.add_argument('--all', action='store_true')
    parser.add_argument(
        '--scores',
        action='store_true',
        help='只打印每题的最高相似度（**不按门过滤**），用来定阈值',
    )
    args = parser.parse_args()

    session_factory = get_session_factory()
    with session_factory() as session:
        entries = list(
            session.execute(
                select(WikiEntry).where(WikiEntry.status == 'reviewed')
            ).scalars()
        )
        by_slug = {entry.slug: entry for entry in entries}
        texts = tuple(
            f'{entry.title} {" ".join(entry.triggers or [])}' for entry in entries
        )
        vectors = _entry_vectors_cached(texts)

        rows = load_rows()
        if not args.all:
            rows = [
                row
                for row in rows
                if pick(row, 'code').startswith(args.only)
            ]

        questions = [pick(row, 'question') for row in rows]
        query_vectors = embed_texts(questions)

        if args.scores:
            # 定阈值要看的是**分布**，不是某一题。
            # 打印时不按门过滤：门是"该不该路由"的判据，阈值是"像不像"的判据，
            # 两个判据要分开标定，否则调一个的时候另一个的变化看不见。
            print(f'{"编号":<7}{"门":<6}{"相似度":<9}词条')
            print('-' * 70)
            for row, query_vector in zip(rows, query_vectors):
                scored = sorted(
                    (
                        (entry, _cosine(query_vector, vector))
                        for entry, vector in zip(entries, vectors)
                    ),
                    key=lambda pair: -pair[1],
                )
                top_entry, top_score = scored[0]
                gate = is_comparison(pick(row, 'question'))
                print(
                    f'{pick(row, "code"):<7}{str(gate):<6}{top_score:<9.3f}{top_entry.slug}'
                )
            return 0

        counts = {'正确': 0, '错误': 0, '未路由': 0, '无词条': 0}
        anchor_miss = 0
        for row, query_vector in zip(rows, query_vectors):
            code = pick(row, 'code')
            scored = sorted(
                (
                    (entry, _cosine(query_vector, vector))
                    for entry, vector in zip(entries, vectors)
                ),
                key=lambda pair: -pair[1],
            )
            top_entry, top_score = scored[0]
            expected_slug = EXPECTED.get(code)
            expected_score = next(
                (score for entry, score in scored if entry.slug == expected_slug), None
            )
            # 门（is_comparison）没过的问题**根本不会走词条**，
            # 所以这里必须照着真实链路判：门没过就是「未路由」，
            # 不能拿"如果过了门它会选中谁"去当结论——那是没发生的事。
            gate = is_comparison(pick(row, 'question'))
            # ⚠️ 关键：这里**调用产品里那个函数**，不自己重算一遍。
            # 自己重算的后果是——产品改了打分方式，诊断还在按老逻辑显示，
            # 于是"诊断说对了、线上是错的"这种最难查的不一致就出现了。
            actual = match_entry_with_vector(
                session, pick(row, 'question'), query_vector=query_vector
            )
            routed = actual.slug if actual is not None else None

            if expected_slug is None:
                verdict = '无词条'
            elif not gate:
                verdict = '未路由'
            elif routed == expected_slug:
                verdict = '正确'
            else:
                verdict = '错误'
            counts[verdict] += 1

            mark = {'正确': '✅', '错误': '❌', '未路由': '🚪', '无词条': '❓'}[verdict]
            print(f'{mark} {code}  {verdict}  门={gate}')
            print(f'     问：{pick(row, "question")}')
            if gate:
                if actual is not None:
                    bonus_len = _keyword_hit(actual, _bare(pick(row, 'question')))
                    semantic = next(
                        score for entry, score in scored if entry.slug == actual.slug
                    )
                    print(
                        f'     实际路由到：{routed}  '
                        f'总分 {semantic + KEYWORD_BONUS_PER_CHAR * bonus_len:.3f} '
                        f'= 语义 {semantic:.3f} + 触发词 '
                        f'{KEYWORD_BONUS_PER_CHAR}×{bonus_len}'
                    )
                else:
                    print(f'     实际路由到：（没路由）')
            print(f'     纯语义最高的是：{top_entry.slug}  {top_score:.3f}')
            print(
                f'     应路由到：{expected_slug or "（无对应词条）"}'
                + (f'  {expected_score:.3f}' if expected_score is not None else '')
            )
            if verdict == '错误':
                print('     ⚠️ 路由挑错了词条')

            # 正确词条的「依据」里到底列了哪些条款？评测锚点在不在这份清单里？
            # 这一栏回答的是**另一个问题**：路由送对了，为什么还是判"未通过"。
            expected_entry = by_slug.get(expected_slug or '')
            if expected_entry is not None:
                cited = {
                    normalize(str(item.get('条款') or ''))
                    for item in (expected_entry.citations or [])
                }
                anchors = [
                    item.strip()
                    for item in pick(row, 'anchor').split(SEPARATOR)
                    if item.strip() and not item.startswith('（')
                ]
                for anchor in anchors:
                    match = ARTICLE.match(normalize(anchor))
                    number = f'第{match.group(1)}条' if match else ''
                    covered = bool(number) and any(number in item for item in cited)
                    if not covered:
                        anchor_miss += 1
                    print(
                        f'     锚点「{anchor[:20]}…」'
                        + ('在词条依据里 ✅' if covered else '不在词条依据里 ⚠️')
                    )

        print()
        print('=' * 78)
        print(
            f'路由正确 {counts["正确"]} ｜ 路由错误 {counts["错误"]} ｜ '
            f'门没过（未路由）{counts["未路由"]} ｜ 无对应词条 {counts["无词条"]}'
        )
        print(f'锚点不在所属词条依据里的次数：{anchor_miss}')
        print()
        print('怎么读这三个数：')
        print('  · 「路由错误」——路由挑错了词条，该改的是路由（阈值/混合判据）；')
        print('  · 「门没过」——问题压根不是"对比词"的写法，该改的是门的判据；')
        print('  · 「锚点不在词条依据里」——路由送对了但词条没写那条依据，')
        print('    该补的是词条，或者**该回头怀疑评测锚点取得太窄**。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
