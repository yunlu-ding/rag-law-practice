"""从语料里挖出"可以拿来出题"的候选句子。

为什么要有这个工具：
评测集要扩大，但**锚点不能凭记忆写**——前面已经栽过两次
（写错条款措辞、写错 `Five-Year`）。手写一百道题的锚点，
出错是必然的，而错误的锚点会让题目永远判"未命中"。

所以反过来做：**让语料自己把候选吐出来**，人只负责挑和改写问法。
脚本保证每一行输出的锚点都是**在语料里真实存在**的（它就是从语料里截的）。

用法：
    cd vibe-rag/backend
    python ../评测/挖候选锚点.py --kind glossary --limit 40
    python ../评测/挖候选锚点.py --kind standards  --limit 40
    python ../评测/挖候选锚点.py --kind tables     --limit 20
    python ../评测/挖候选锚点.py --kind numeric    --limit 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402


def load_chunks(db, filename_like: str) -> list[Chunk]:
    document = db.execute(
        select(Document).where(Document.filename.like(f'%{filename_like}%'))
    ).scalars().first()
    if document is None:
        return []
    return list(
        db.execute(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
        ).scalars().all()
    )


def clean(text: str) -> str:
    """把 PDF 提取出来的排版噪音压掉，但**不修改锚点原文**。

    这里只做"显示用"的清洗：把换行压成空格、多个空格压成一个。
    锚点本身必须是从语料里原样截出来的连续片段——
    一旦我们"顺手修一下"，锚点就可能不再存在于语料里，题目就废了。
    """

    return re.sub(r'\s+', ' ', text).strip()


# 术语表里每个词条的样子：一个首字母大写的词组，紧跟一句以 The/A/An/In/When 开头的定义。
GLOSSARY_ENTRY = re.compile(
    r'([A-Z][A-Za-z\-]+(?: [A-Za-z\-]+){0,4}) '
    r'((?:The|A|An|In|When|For|Any|Two|T he) [^.]{35,190}\.)'
)

# 准则的标题形态，例如 "III(B) Fair Dealing"
STANDARD_HEADING = re.compile(r'\b([IV]{1,3}\([A-Z]\))\s+([A-Z][A-Za-z ,\-]{4,40})')

# 表格：Exhibit 是这本书里表格的统一标记
EXHIBIT = re.compile(r'Exhibit\s+\d+')

# 数值型句子：含百分比或明确数字
NUMERIC = re.compile(r'[^.]{25,180}?\d+(?:\.\d+)?\s?(?:percent|%|\bfund\b|euros?)[^.]{0,60}\.')


def main() -> int:
    parser = argparse.ArgumentParser(description='从语料里挖候选锚点')
    parser.add_argument('--kind', required=True, choices=['glossary', 'standards', 'tables', 'numeric'])
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()

    with get_session_factory()() as db:
        if args.kind == 'glossary':
            chunks = load_chunks(db, 'glossary')
            pattern, label = GLOSSARY_ENTRY, '术语表候选（词组 + 定义）'
        elif args.kind == 'standards':
            chunks = load_chunks(db, 'standards-practice-handbook')
            pattern, label = STANDARD_HEADING, '准则标题候选'
        elif args.kind == 'tables':
            chunks = load_chunks(db, 'Quantitive Methods')
            pattern, label = EXHIBIT, '表格（Exhibit）候选'
        else:
            chunks = load_chunks(db, 'Quantitive Methods')
            pattern, label = NUMERIC, '数值句候选'

    print(f'===== {label} ｜ 语料 {len(chunks)} 片 =====\n')

    seen: set[str] = set()
    count = 0
    for chunk in chunks:
        text = clean(chunk.content)
        for match in pattern.finditer(text):
            if args.kind == 'glossary':
                term, definition = match.group(1), match.group(2)
                key = term.lower()
                if key in seen or len(term) < 6:
                    continue
                seen.add(key)
                print(f'[{count:03d}] 术语: {term}')
                print(f'      锚点: {definition[:120]}')
                print(f'      第 {chunk.page_number} 页 ｜ 切片 {chunk.chunk_index}')
            elif args.kind == 'standards':
                code, name = match.group(1), match.group(2).strip()
                key = code
                if key in seen:
                    continue
                seen.add(key)
                # 取标题之后的那段作为候选锚点
                tail = text[match.end():match.end() + 150].strip()
                print(f'[{count:03d}] 条款: {code} {name}')
                print(f'      后续原文: {tail[:120]}')
                print(f'      第 {chunk.page_number} 页 ｜ 切片 {chunk.chunk_index}')
            elif args.kind == 'tables':
                key = match.group(0)
                if key in seen:
                    continue
                seen.add(key)
                start = max(match.start() - 40, 0)
                print(f'[{count:03d}] 片段: {text[start:match.end() + 150]}')
                print(f'      第 {chunk.page_number} 页 ｜ 切片 {chunk.chunk_index}')
            else:
                sentence = match.group(0).strip()
                if sentence[:60] in seen:
                    continue
                seen.add(sentence[:60])
                print(f'[{count:03d}] 数值句: {sentence[:170]}')
                print(f'      第 {chunk.page_number} 页 ｜ 切片 {chunk.chunk_index}')
            print()
            count += 1
            if count >= args.limit:
                print(f'（已达上限 {args.limit} 条）')
                return 0

    print(f'（共 {count} 条候选）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
