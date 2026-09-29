"""把国务院公报里"一文多规章"的页面拆成一份份独立的语料文件。

为什么需要拆：

国务院公报的页面结构是"**一份令 + 若干附件**"：

    中国证券监督管理委员会令（第202号）关于修改、废止部分证券期货规章的决定
      附件1  决定修改的规章（8 部）
        一、将《A办法》第一条修改为：……      ← 修改说明
        A办法                                 ← 该规章的合并文本（重新公布）
        B办法
        ……
      附件2  决定废止的规章

如果整份入库，会产生两个问题：

  1. **引用会指错。** 用户问《证券期货投资者适当性管理办法》第三十九条，
     检索到的切片来自这份公报，引用标签只能写"证监会令第202号"，
     写不出"《适当性管理办法》"——因为在那份文档里，它们确实只是附件的一部分。
  2. **多部规章混在一个文档里，条号会互相打架。**
     8 部规章各自都有"第一条""第二条"，切片正文里分不清是谁的第几条。

所以先按规章拆开，每部规章成为一个独立的语料文件——
文件名就是它的身份，引用标签自然就对了。

拆分依据（不靠猜）：
  - 合并文本的起点，是"《…》等N部规章根据本决定作相应修改，重新公布。"这句话；
  - 之后每部规章的边界是"**一行标题 + 紧随其后一行以"（YYYY年"开头**"——
    标题是规章全称，括号里是它的通过/修正沿革，这是官方文本的固定写法；
  - 终点是"附件2"。

用法：
    python 工具/拆分公报.py             # 演练，只列出会生成什么
    python 工具/拆分公报.py --apply     # 真正写出文件
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
logging.getLogger('pypdf').setLevel(logging.ERROR)

from app.rag.loader import load_document  # noqa: E402

CORPUS = ROOT / '语料'
OUTPUT_DIR = CORPUS / '部门规章'

# "《…》等N部规章根据本决定作相应修改，重新公布。" —— 合并文本的起点。
REPUBLISH = re.compile(r'规章根据本决定作相应修改，重新公布')

# 沿革说明行：以全角/半角括号 + 四位年份开头。
HISTORY = re.compile(r'^[（(]\s*(19|20)\d{2}\s*年')

# 标题行：规章全称。用长度和结尾词约束，避免把正文里的句子当成标题。
TITLE_TAIL = re.compile(r'(办法|规定|细则|指引|规则|准则|决定)$')


def find_gazette() -> Path:
    for path in CORPUS.iterdir():
        if path.is_file() and '202' in path.name and path.suffix.lower() == '.html':
            return path
    raise FileNotFoundError('语料目录里没找到公报文件')


def normalize(title: str) -> str:
    return '《' + title.strip().strip('《》') + '》'


def safe_filename(title: str) -> str:
    """把规章名变成文件名。

    去掉文件名里不能用的字符，并把书名号去掉——文件名里带书名号
    在 Windows 上虽然合法，但会让路径看起来很奇怪，而且命令行里要转义。
    """

    cleaned = title.strip().strip('《》')
    cleaned = re.sub(r'[\\/:*?"<>|]', '', cleaned)
    return cleaned.strip()


def split_gazette(paragraphs: list[str]) -> tuple[str, list[tuple[str, list[str]]], list[str]]:
    """返回 (决定正文, [(规章名, 段落列表)], 附件2段落)。"""

    republish_index = next(
        (index for index, line in enumerate(paragraphs) if REPUBLISH.search(line)), None
    )
    if republish_index is None:
        raise ValueError('没找到"重新公布"这句话，公报结构可能变了')

    appendix2_index = next(
        (
            index
            for index, line in enumerate(paragraphs)
            if index > republish_index and re.match(r'^附件2', line)
        ),
        len(paragraphs),
    )

    decision_lines = paragraphs[: republish_index + 1]
    tail_lines = paragraphs[appendix2_index:]

    # 找每部规章的起点：标题行的下一行是沿革说明。
    starts: list[tuple[int, str]] = []
    for index in range(republish_index + 1, appendix2_index - 1):
        title = paragraphs[index].strip()
        if not title or len(title) > 60:
            continue
        if not TITLE_TAIL.search(title.strip('《》')):
            continue
        if HISTORY.match(paragraphs[index + 1].strip()):
            starts.append((index, title))

    documents: list[tuple[str, list[str]]] = []
    for order, (start, title) in enumerate(starts):
        end = starts[order + 1][0] if order + 1 < len(starts) else appendix2_index
        documents.append((title, paragraphs[start:end]))

    return '\n'.join(decision_lines), documents, tail_lines


def main() -> int:
    parser = argparse.ArgumentParser(description='拆分国务院公报')
    parser.add_argument('--apply', action='store_true', help='真正写出文件')
    args = parser.parse_args()

    gazette = find_gazette()
    loaded = load_document(gazette)
    paragraphs = [line for line in loaded.full_text.split('\n') if line.strip()]

    decision, documents, appendix2 = split_gazette(paragraphs)

    print(f'来源：{gazette.name[:50]}…')
    print(f'总段落：{len(paragraphs)}')
    print()
    print(f'决定正文：{len(decision)} 字')
    print()
    print(f'拆分出 {len(documents)} 部规章：')
    for title, lines in documents:
        body = '\n'.join(lines)
        print(f'  {safe_filename(title):<34} {len(body):>6} 字  {len(lines):>4} 段')
    print()
    print(f'附件2（废止清单）：{len(chr(10).join(appendix2))} 字')

    if not args.apply:
        print()
        print('演练模式：什么都没写。确认要落盘就加 --apply。')
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 决定正文 + 附件2 合成一份：它们是"这份令改了什么、废止了什么"的记录，
    # 拆开反而不好引用。
    decision_path = OUTPUT_DIR / '证监会令第202号_关于修改废止部分证券期货规章的决定.txt'
    decision_path.write_text(
        '关于修改、废止部分证券期货规章的决定\n\n'
        + decision
        + '\n\n'
        + '\n'.join(appendix2),
        encoding='utf-8',
    )
    written.append(decision_path)

    for title, lines in documents:
        path = OUTPUT_DIR / f'{safe_filename(title)}.txt'
        path.write_text('\n'.join(lines), encoding='utf-8')
        written.append(path)

    print()
    for path in written:
        print(f'  已写出 {path.name}')
    print()
    print(f'共 {len(written)} 个文件，都在 {OUTPUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
