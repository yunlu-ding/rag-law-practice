from __future__ import annotations

import re

from app.rag.splitters.base import (
    DEFAULT_CHUNK_SIZE,
    HARD_LIMIT_MULTIPLIER,
    MIN_CHUNK_SIZE,
    SplitChunk,
    find_best_split_point,
)

"""结构感知切分。

核心洞察来自一个很朴素的观察：
**真实文档几乎都有作者亲手划好的结构边界。**

- 法律条文：`第一条`、`Article 3`
- 行业标准：`Standard III(B)`
- 考试教材：`A. Knowledge of the Law`
- 技术文档：Markdown 标题

这些标记比任何"按长度猜"的算法都准，而且它们不针对某一种文档——
换成合同、法规、手册同样适用。

所以策略是三级降级：
    ① 按结构标记切      ← 最准
    ② 句子边界切        ← 次优
    ③ 空格边界切        ← 兜底，绝不从单词中间切断
"""

# 强边界：这些是明确的一级结构，绝不允许被合并掉。
# 一旦跨过强边界合并，几条独立规则就被揉进同一片，引用无法精确到某一条。
STRONG_HEADING_PATTERNS = [
    re.compile(r'^#{1,6}\s+\S'),                                     # Markdown 标题
    re.compile(r'^第[一二三四五六七八九十百千]+[条章节编]'),               # 中文条款
    re.compile(r'^Article\s+\d+', re.IGNORECASE),                     # 英文法条
    re.compile(r'^Standard\s+[IVX]+(\s*\([A-Z]\))?\b'),               # 行业标准编号
    re.compile(r'^[IVX]{1,5}\.\s+[A-Z]'),                             # 罗马数字小标题
    re.compile(r'^\d+\.\s+[A-Z]'),                                    # 数字小标题
    re.compile(r'^[A-Z][A-Z0-9 ,&/\'\-]{8,}$'),                       # 全大写标题行
]

# 弱边界：看起来像小标题，但也可能只是正文里的编号句，允许为控制切片数量而合并。
WEAK_HEADING_PATTERNS = [
    re.compile(r'^[A-Z]\.\s+\S'),                                     # A. xxx
    re.compile(r'^\d+\.\d+\s+\S'),                                    # 1.2 xxx
    re.compile(r'^[a-z]\)\s+\S'),                                     # a) xxx
]

# 标题不会很长。超过这个长度的一律当成正文，
# 否则正文里随便一句以数字开头的句子都会被误判成小标题。
HEADING_MAX_LENGTH = 110


def _heading_level(line: str) -> str:
    """判断一行是不是结构标记，返回 'strong' / 'weak' / ''。"""

    stripped = line.strip()
    if not stripped or len(stripped) > HEADING_MAX_LENGTH:
        return ''
    for pattern in STRONG_HEADING_PATTERNS:
        if pattern.match(stripped):
            return 'strong'
    for pattern in WEAK_HEADING_PATTERNS:
        if pattern.match(stripped):
            return 'weak'
    return ''


def looks_structured(text: str, *, sample_lines: int = 400, min_headings: int = 3) -> bool:
    """判断这段文本有没有结构。

    只看前若干行，因为判断"有没有结构"不需要读完全文，
    而长文档读全文会很慢。
    """

    if not text:
        return False
    headings = 0
    for line in text.split('\n')[:sample_lines]:
        if _heading_level(line):
            headings += 1
            if headings >= min_headings:
                return True
    return headings >= min_headings


def _iter_units(text: str) -> list[tuple[int, int, bool]]:
    """把文本按结构标记切成"单元"，返回 (起点, 终点, 是否强边界) 列表。"""

    units: list[tuple[int, int, bool]] = []
    cursor = 0
    unit_start = 0
    unit_is_strong = False
    has_body = False
    started = False

    for line in text.split('\n'):
        line_start = cursor
        cursor += len(line) + 1
        level = _heading_level(line) if line.strip() else ''
        if not started:
            started = True
            unit_start = line_start
            unit_is_strong = (level == 'strong')
            has_body = bool(line.strip()) and not level
            continue
        if level:
            if has_body:
                # 前一个单元已经有正文，这一行开启新单元
                units.append((unit_start, line_start, unit_is_strong))
                unit_start = line_start
                unit_is_strong = (level == 'strong')
                has_body = False
            else:
                # 连续的多行：把它当成一个"多行标题"并进同一个单元。
                # 为什么要这样处理：PDF 提取出来的标题经常被排版拆成好几行，
                # 例如 "CODE OF ETHICS" / "AND STANDARDS OF" / "PROFESSIONAL CONDUCT"，
                # 按行切会得到三个十几字的碎片。
                unit_is_strong = unit_is_strong or (level == 'strong')
        elif line.strip():
            has_body = True

    if started:
        units.append((unit_start, len(text), unit_is_strong))
    return units


def _merge_units(
    text: str,
    units: list[tuple[int, int, bool]],
    chunk_size: int,
) -> list[tuple[int, int]]:
    """把相邻的小单元合并，但不跨强边界。

    合并的目的是避免产生一堆十几个字的碎片切片——
    碎片切片在检索时几乎必然被噪声淹没。

    "不跨强边界合并"这条规则保护的是：**不要把两条独立规则揉进同一片**，
    否则引用就无法精确到某一条。

    但它有一个例外：当某一侧短到根本不携带信息时（比如一行页脚、
    一个被排版拆出来的半截标题），合并并不会造成"两条规则混在一起"的后果，
    反而能避免碎片占着检索名额。所以这里的判断加了一个最小长度条件。
    """

    merged: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    current_strong = False

    for start, end, strong in units:
        if current_start is None:
            current_start, current_end, current_strong = start, end, strong
            continue
        can_merge = (
            (end - current_start) <= chunk_size
            and (
                (not current_strong and not strong)
                or (current_end - current_start) < MIN_CHUNK_SIZE
                or (end - start) < MIN_CHUNK_SIZE
            )
        )
        if can_merge:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end, current_strong = start, end, strong

    if current_start is not None:
        merged.append((current_start, current_end))
    return merged


def split_semantic_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    hard_limit_multiplier: float = HARD_LIMIT_MULTIPLIER,
) -> list[SplitChunk]:
    """按文档自身的结构边界切分。"""

    if not text:
        return []

    hard_limit = int(chunk_size * hard_limit_multiplier)
    units = _iter_units(text)
    merged = _merge_units(text, units, chunk_size)

    # 先只算出"切哪些区间"，最后再统一从原文取内容。
    # 这样做的好处是每个切片的内容严格等于原文的一段——
    # 不做字符串拼接，就不会在合并过程中悄悄改变原文（比如漏掉一个换行）。
    ranges: list[tuple[int, int]] = []
    for start, end in merged:
        # 语义完整优先：没超过硬上限就整片保留，哪怕比目标长度长。
        if end - start <= hard_limit:
            ranges.append((start, end))
            continue

        position = start
        while (end - position) > hard_limit:
            cut = find_best_split_point(text, position, chunk_size)
            cut = min(cut, end)
            if cut <= position:
                cut = min(position + chunk_size, end)
            ranges.append((position, cut))
            position = cut
        if position < end:
            ranges.append((position, end))

    ranges = _absorb_tiny_ranges(ranges)

    # 去掉纯空白切片：它们不携带信息，只会污染检索
    return [
        SplitChunk(content=text[start:end], start_offset=start, end_offset=end)
        for start, end in ranges
        if text[start:end].strip()
    ]


def _absorb_tiny_ranges(
    ranges: list[tuple[int, int]],
    *,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """把过短的区间并进相邻区间。

    为什么这一步要单独放在最后，而不是在合并阶段顺手做掉：
    合并阶段受"不超过目标长度"约束。当一片 42 字的页脚后面跟着一个
    1296 字的正文单元时，合并会被长度限制挡回去，碎片就留下来了。

    所以收尾再扫一遍：**碎片不携带可检索的信息，让它独立存在的唯一后果
    就是占掉一个检索名额，把有用的内容挤出去。**
    """

    if not ranges:
        return ranges

    result: list[tuple[int, int]] = []
    for start, end in ranges:
        if result and (end - start) < min_size:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))

    # 第一片如果本身太短，并进后面那一片（它前面没有邻居了）
    if len(result) > 1 and (result[0][1] - result[0][0]) < min_size:
        first = result.pop(0)
        result[0] = (first[0], result[0][1])

    return result


__all__ = ['split_semantic_text', 'looks_structured']
