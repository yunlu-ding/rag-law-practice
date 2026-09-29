"""从语料里逐字抽取评测集的锚点原文。

为什么锚点必须由脚本生成，而不是手工从法规里抄：

**判定"检索有没有命中"靠的是逐字比对。** 手抄的时候人很容易顺手把
标点改掉（把"，"写成"、"），或者在换行处多补一个空格——
这些在阅读时毫无差别，但比对时会**全部判为未命中**，
于是评测报告显示"检索全线失败"，而真实原因只是锚点抄错了一个字。

这个脚本从数据库里取切片正文，按（文件 + 条款号）精确定位，
截取前 N 个字作为锚点。它同时是一道**校验**：
如果某个（文件，条款）组合找不到，说明评测集里这一行写错了。

用法：
    python 评测/生成锚点.py            # 演练，只报告能不能定位
    python 评测/生成锚点.py --apply    # 写回 CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '评测'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.rag.splitters.legal import chinese_number_to_int  # noqa: E402
from 评测集 import ANCHOR_COLUMN, load_rows, pick, save_rows  # noqa: E402

ANCHOR_CHARS = 36
SEPARATOR = '｜'

# 这些取值表示"这一行本来就该没有锚点"，不是错误。
NO_ANCHOR_MARKERS = ('语料不覆盖', '语料中该条不存在', '语料只含法规原文', '超出知识库范围')


def load_corpus(session) -> dict[str, dict]:
    """把语料读成 {文件名: {'chunks': [...], 'preamble': str}}。"""

    corpus: dict[str, dict] = {}
    for document in session.execute(select(Document).where(Document.status == 'indexed')).scalars():
        chunks = list(
            session.execute(
                select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
            ).scalars()
        )
        by_article: dict[int, str] = {}
        for chunk in chunks:
            number = chinese_number_to_int(chunk.article_number or '')
            if number is None or number in by_article:
                # 同一条可能有续片；锚点用**第一片**（条文头），
                # 因为检索回来看的就是那一片。
                continue
            by_article[number] = chunk.content
        corpus[document.filename] = {
            'by_article': by_article,
            'first_chunk': chunks[0].content if chunks else '',
            'all_text': '\n'.join(chunk.content for chunk in chunks),
        }
    return corpus


def cut(text: str) -> str:
    """取前 N 个可见字符，并压掉换行——锚点要能在一行里读。"""

    flat = ' '.join(text.split())
    return flat[:ANCHOR_CHARS]


def anchor_for(corpus: dict[str, dict], filename: str, article: str) -> tuple[str | None, str]:
    """返回 (锚点, 说明)。锚点为 None 表示定位失败。"""

    entry = corpus.get(filename)
    if entry is None:
        return None, f'语料里没有这份文件：{filename}'

    article = article.strip()

    if article.startswith('第') and article.endswith('条'):
        number = chinese_number_to_int(article)
        text = entry['by_article'].get(number)
        if text is None:
            return None, f'{filename} 里没有 {article}'
        return cut(text), f'{article} 已定位'

    if article in ('沿革说明', '首段'):
        # 沿革说明和首段都是文档开头的"前言"部分。
        if article == '沿革说明':
            marker = entry['all_text'].find('根据2022年8月12日')
            if marker >= 0:
                return cut(entry['all_text'][marker:]), '沿革说明已定位（2022年修正）'
        if entry['first_chunk']:
            return cut(entry['first_chunk']), f'{article} 取文档开头'
        return None, f'{filename} 没有内容'

    return None, f'无法识别的条款写法：{article}'


def main() -> int:
    parser = argparse.ArgumentParser(description='生成评测集锚点')
    parser.add_argument('--apply', action='store_true', help='把锚点写回 CSV')
    args = parser.parse_args()

    session_factory = get_session_factory()
    with session_factory() as session:
        corpus = load_corpus(session)

    rows = load_rows()

    filled = skipped = failed = 0
    problems: list[str] = []

    for row in rows:
        source = pick(row, 'source_file').strip()
        if any(marker in source for marker in NO_ANCHOR_MARKERS) or source in ('', '，'):
            skipped += 1
            row[ANCHOR_COLUMN] = pick(row, 'anchor') or '（本题无锚点）'
            continue

        filenames = [item.strip() for item in source.split(SEPARATOR)]
        articles = [item.strip() for item in pick(row, 'source_article').split(SEPARATOR)]
        # 一行里写了同一份文件的好几个条款（比如"第七条｜第二十三条"）时，
        # 文件列只写一次就够了，不必把同一个文件名抄两遍。
        if len(filenames) == 1 and len(articles) > 1:
            filenames = filenames * len(articles)
        if len(filenames) != len(articles):
            failed += 1
            problems.append(
                f"{pick(row, 'code')}：文件数（{len(filenames)}）和条款数（{len(articles)}）对不上"
            )
            continue

        anchors: list[str] = []
        row_ok = True
        for filename, article in zip(filenames, articles):
            anchor, message = anchor_for(corpus, filename, article)
            if anchor is None:
                failed += 1
                row_ok = False
                problems.append(f"{pick(row, 'code')}：{message}")
                continue
            anchors.append(anchor)

        if row_ok:
            row[ANCHOR_COLUMN] = SEPARATOR.join(anchors)
            filled += 1
            print(f"✅ {pick(row, 'code'):<7} {row[ANCHOR_COLUMN][:56]}")
        else:
            print(f"❌ {pick(row, 'code'):<7} 定位失败")

    print()
    print(f'已生成锚点 {filled} 题 ｜ 无锚点（拒答/说明类）{skipped} 题 ｜ 失败 {failed} 题')
    if problems:
        print()
        print('需要人工处理的：')
        for problem in problems:
            print(f'  · {problem}')

    if not args.apply:
        print()
        print('演练模式：没有写回文件。确认后加 --apply。')
        return 0 if not problems else 1

    save_rows(rows)
    print()
    print('已写回评测集')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
