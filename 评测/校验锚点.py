"""锚点校验：确认评测集里的每条锚点原文**真的存在于语料里**。

为什么必须先做这一步：
评测集里的"锚点"是用来判定检索是否命中的。如果锚点本身在语料里根本不存在，
那么这条题永远不可能命中——但它看起来只是"系统没检索到"，
会让人误以为是检索能力不行，去改一个根本没问题的地方。

**错误的评测集比没有评测集更糟**：它会把优化引向错误的方向。

这个脚本不需要联网，直接查数据库里的切片，所以可以反复跑。

用法：
    cd vibe-rag/backend
    python ../评测/校验锚点.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from sqlalchemy import select  # noqa: E402

from app.core.postgres import get_session_factory  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / '评测集.csv'


def normalize(text: str) -> str:
    """归一化，用于锚点匹配。

    材料是从 PDF 抽出来的，里面的引号可能是弯引号、破折号可能是长破折号、
    换行会把短语切断。所以匹配之前要**去掉所有标点、统一空白**，
    只比较字母和数字——否则会因为一个引号差异判定为"没命中",
    而这种假失败最容易被误读成"检索不行"。
    """

    lowered = str(text or '').lower()
    lowered = lowered.replace('’', "'").replace('‘', "'")
    lowered = lowered.replace('“', '"').replace('”', '"')
    # 保留字母、数字、空格；其余全部去掉
    cleaned = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', lowered)
    return re.sub(r'\s+', ' ', cleaned).strip()


def main() -> int:
    rows = list(csv.DictReader(EVAL_SET.open(encoding='utf-8')))
    print(f'评测集共 {len(rows)} 条')

    session_factory = get_session_factory()
    with session_factory() as db:
        chunks = list(db.execute(select(Chunk).where(Chunk.enabled.is_(True))).scalars().all())

    normalized_corpus = [normalize(chunk.content) for chunk in chunks]
    print(f'语料切片 {len(chunks)} 条，已归一化\n')

    checked = 0
    failures: list[tuple[str, str]] = []

    for row in rows:
        anchor = (row.get('锚点原文') or '').strip()
        if not anchor:
            continue
        checked += 1
        target = normalize(anchor)
        hit_index = next(
            (i for i, text in enumerate(normalized_corpus) if target in text),
            None,
        )
        if hit_index is None:
            failures.append((row['编号'], anchor))
            print(f'  ✗ {row["编号"]:<8} 锚点不存在: {anchor}')
        else:
            chunk = chunks[hit_index]
            print(
                f'  ✓ {row["编号"]:<8} 命中切片 {chunk.chunk_index:<5}'
                f' 第{chunk.page_number}页  {anchor[:56]}'
            )

    print()
    print('=' * 76)
    print(f'校验了 {checked} 条锚点，失败 {len(failures)} 条')
    if failures:
        print('下面这些锚点需要修正（它们会让对应题目永远无法命中）：')
        for code, anchor in failures:
            print(f'  {code}: {anchor}')
        return 1
    print('→ 全部锚点均在语料中，可以开始评测')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
