"""统计法规之间的引用边——决定"要不要建关系层"之前先量一下。

为什么要先量：

关系层有两种边，成本天差地别。

    引用边（A 条引用了 B 条）  —— 法规里到处是"按照《X》第N条"，**正则能抽**
    平行边（A 条和 B 条讲同一件事）—— 只能人判，和写 Wiki 词条是同一类人工活

如果自动抽出来的引用边本身就够用（比如能支撑跨层级综合类问题），
那最贵的那半只需要覆盖少数高频对比——这套方案就划算。
**所以先量的不是平行边，是引用边的产量和质量。**

量的四个维度（对应决策）：

  1. 引用边总数        —— 法规互相引用的密度够不够
  2. 跨层级引用数      —— 能不能支撑"跨层级综合"
  3. 条款级引用数      —— 能不能精确到条（只到法规名的边用处小得多）
  4. 方向分布          —— 下位→上位是"依据/细化"，同位是"交叉引用"，
                          上位→下位是"授权/委托"。**引用不等于上下位**，
                          不区分方向，边就只能表示"提过"，表示不了"关系"。

第 5 个维度（**人工核对准确率**）机器量不了，脚本会抽样打出来，
由人看几十条确认。

用法：
    python 工具/统计引用边.py
    python 工具/统计引用边.py --sample 30
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.rag.metadata import level_label  # noqa: E402
from app.rag.splitters.legal import chinese_number_to_int  # noqa: E402

# 带书名号、精确到条的引用：按照《证券法》第一百九十八条
_CITED_ARTICLE = re.compile(r'《([^》]{2,40})》\s*第([一二三四五六七八九十百零〇\d]{1,8})条')
# 只提到法规、没到条：《证券法》的规定
_CITED_DOC_ONLY = re.compile(r'《([^》]{2,40})》(?!\s*第[一二三四五六七八九十百零〇\d]{1,8}条)')
# 同文档内引用：本办法第二十九条 / 本指引第十二条
_SELF_ARTICLE = re.compile(r'本(?:办法|指引|细则|规定|条例|法)第([一二三四五六七八九十百零〇\d]{1,8})条')

# 引用前面的引导词。它决定这条边是"依据"还是"违反"——
# 两者在图上方向相同，但语义完全不同（一个是遵循，一个是被处罚的对照）。
_LEADING_WORDS = ('按照', '依据', '根据', '参照', '符合', '违反', '遵守', '适用', '见')


def _clean(name: str) -> str:
    return re.sub(r'[\s\u3000]+', '', name).strip('《》')


def _match_document(name: str, documents: list[dict]) -> dict | None:
    """把引用里写的法规名匹配到语料里的某份文件。

    用后缀匹配：法规常被简称（"证券法"之于"中华人民共和国证券法"），
    而简称总是全称的后缀。
    """

    cleaned = _clean(name)
    best: tuple[int, dict] | None = None
    for document in documents:
        title = document['title']
        for length in range(len(title), 2, -1):
            if title[-length:] in cleaned:
                if best is None or length > best[0]:
                    best = (length, document)
                break
    return best[1] if best else None


def _leading_word(text: str, position: int) -> str:
    window = text[max(0, position - 6) : position]
    for word in _LEADING_WORDS:
        if word in window:
            return word
    return ''


def main() -> int:
    parser = argparse.ArgumentParser(description='统计法规引用边')
    parser.add_argument('--sample', type=int, default=20, help='抽样多少条给人核对')
    parser.add_argument(
        '--penalty',
        nargs=2,
        metavar=('法规关键字', '条号'),
        default=None,
        help='查"这条违反了按哪条罚"：沿引用边找它指向的罚则条款',
    )
    args = parser.parse_args()

    session_factory = get_session_factory()
    with session_factory() as session:
        documents = [
            {
                'id': d.id,
                'title': d.title or d.filename,
                'rank': d.level_rank or 0,
                'level': d.legal_level,
            }
            for d in session.execute(select(Document).where(Document.status == 'indexed')).scalars()
        ]
        chunks = list(session.execute(select(Chunk).order_by(Chunk.chunk_index)).scalars())

    by_id = {d['id']: d for d in documents}

    article_edges: list[dict] = []
    doc_only = 0
    self_article = 0
    unresolved_doc = Counter()
    unresolved_article = 0

    for chunk in chunks:
        text = chunk.content or ''
        source = by_id.get(chunk.document_id)
        if source is None:
            continue

        for match in _CITED_ARTICLE.finditer(text):
            target = _match_document(match.group(1), documents)
            if target is None:
                unresolved_doc[match.group(1)] += 1
                continue
            number = chinese_number_to_int('第' + match.group(2) + '条')
            if number is None:
                continue
            article_edges.append(
                {
                    'source': source,
                    # 引用方是**哪一条**。少了它，边只能表示"A 法规提到过 B 法规"，
                    # 表示不了"某一条的罚则指向某一条"——
                    # 而后者才是这套边唯一能立刻兑现的用途。
                    'source_article': chunk.article_number or '（前言）',
                    'target': target,
                    'article': f'第{match.group(2)}条',
                    'word': _leading_word(text, match.start()),
                    'raw': match.group(0),
                }
            )

        doc_only += len(_CITED_DOC_ONLY.findall(text))
        self_article += len(_SELF_ARTICLE.findall(text))

    # ---- 汇总 ----
    print('=' * 78)
    print('引用边统计')
    print('=' * 78)
    print(f'语料：{len(documents)} 份法规，{len(chunks)} 个切片')
    print()
    print(f'条款级引用边（精确到"第N条"）：{len(article_edges)} 条')
    print(f'只到法规名、没到条：{doc_only} 处（用处小得多，不计入边）')
    print(f'同文档内引用（"本办法第N条"）：{self_article} 处')

    if article_edges:
        cross = [e for e in article_edges if e['source']['rank'] != e['target']['rank']]
        same = [e for e in article_edges if e['source']['rank'] == e['target']['rank']]
        print()
        print(f'跨层级引用：{len(cross)} 条 ｜ 同层级引用：{len(same)} 条')

        print()
        print('跨层级的方向分布（方向决定这条边是"依据"还是"授权"）：')
        direction = Counter()
        for edge in cross:
            src, tgt = edge['source']['rank'], edge['target']['rank']
            if src > tgt:
                direction[f'下位 → 上位（依据/细化）'] += 1
            else:
                direction[f'上位 → 下位（授权/委托）'] += 1
        for name, count in direction.most_common():
            print(f'  {name}：{count} 条')

        print()
        print('引导词分布（决定边的语义）：')
        for word, count in Counter(e['word'] or '（无引导词）' for e in article_edges).most_common(8):
            print(f'  {word}：{count} 条')

        print()
        print('被引用最多的目标（前 8）：')
        for target, count in Counter(
            f"{e['target']['title']} {e['article']}" for e in article_edges
        ).most_common(8):
            print(f'  {count:>3} 次  {target}')

        if args.sample > 0:
            print()
            print(f'⚠️ 人工核对抽样（{args.sample} 条）——这一步机器判不了：')
            step = max(1, len(article_edges) // args.sample)
            for edge in article_edges[::step][: args.sample]:
                print(
                    f"  {edge['source']['title'][:16]:<18} → "
                    f"{edge['target']['title'][:18]:<20} {edge['article']:<10} "
                    f"[{edge['word'] or '—'}] {edge['raw']}"
                )
    if unresolved_doc:
        print()
        print('引用了语料里没有的法规（这些边抽不出来）：')
        for name, count in unresolved_doc.most_common(8):
            print(f'  {count:>3} 次  《{name}》')

    # ---- 用这 75 条边做一件立刻有价值的查询 ----
    #
    # 统计发现引用边几乎全是"罚则引用"：某条违反了什么，按哪一条处罚。
    # 那就先把这个语义兑现出来——它是**现成的边、密度最高、而且不用标平行边**。
    if args.penalty:
        keyword, article = args.penalty
        current = [
            edge
            for edge in article_edges
            if keyword in edge['source']['title']
            and article in str(edge['source_article'])
        ]
        print()
        print('=' * 78)
        print(f'查询：{keyword} {article} 违反了按哪条罚')
        print('=' * 78)
        if not current:
            print('  没有从这个条款出发的引用边。')
            print('  （可能是：该条本身不涉及罚则，或者它是被引用方而不是引用方）')
        for edge in current:
            print(f"  → 《{edge['target']['title']}》{edge['article']}")
            print(f"    引导词：{edge['word'] or '（无）'}　原文片段：{edge['raw']}")

    print()
    print('判据（来自方案）：')
    print('  · 跨层级引用边数量够不够覆盖跨层级类测试题；')
    print('  · 自动抽取准确率 ≥ 90%（要人看上面那批抽样）；')
    print('  · 平行边只需覆盖少数高频对比。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
