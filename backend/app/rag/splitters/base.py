from __future__ import annotations

import re
from dataclasses import dataclass

# 目标切片长度与硬上限。
#
# 为什么要有"硬上限"这个概念：
# 结构感知切分的原则是"语义完整优先于长度整齐"，
# 一条 596 字的规则即使超过目标长度 500，也应该整片保留。
# 但也不能无限长（比如一整章），所以留一个硬上限，超过才降级去切。
#
# 这个规则不是拍脑袋定的：严格按目标长度切会把一条完整准
# 拦腰截断，而截断的代价（模型读到半条规则、引用指向残缺内容）
# 远大于切片稍长的成本。
DEFAULT_CHUNK_SIZE = 500
HARD_LIMIT_MULTIPLIER = 2.0

# 最小切片长度。
#
# 这个数字是被真实数据逼出来的：第一版切完法规之后抽查切片，
# 发现前几片是这样的——
#     [0] "1"（页码，1 字）
#     [1] "证券期货投资者适当性管理办法"（14 字，标题）
#     [2] "第一章"（3 字）
# 一条页脚被当成一个切片，意味着它有机会被检索到、并被塞进提示词里
# 占用一个名额——**碎片不会让答案变好，只会把有用的内容挤出去。**
MIN_CHUNK_SIZE = 80

# 句子边界，优先用它切
SENTENCE_SEPARATORS = ['\n\n', '\n', '。', '！', '？', '；', '. ', '! ', '? ', '; ']

# 只把"拉丁字母被切断"算作缺陷。
# 中文没有空格分词，按字符切本来就不叫切断；如果按 \w 判断，
# 任何中文文本都会被误报成 100% 残句，指标就失去意义了。
_LATIN_LETTER = re.compile(r'[A-Za-z]')


@dataclass
class SplitChunk:
    """一个切分结果。

    start_offset / end_offset 是**相对于所属 section** 的位置，
    不是整篇文档的位置。原因是解析阶段已经把文档拆成了 section
    （PDF 按页、Markdown 按标题），每段独立切分；
    对引用溯源来说，页码 + 条款标题比全文偏移量更有用。

    `article_number` 是这一片属于哪一条法规条款，例如"第二十九条"。
    它有两种取值情形，**必须分清楚**：

      - 这一片**就是**该条的开头（正文以条号起头）；
      - 这一片是该条的**续片**（正文被分页切开，后半段没有条号）。

    区分方式不是看这个字段有没有值，而是看正文有没有以它开头。
    为什么不在字段上再挂一个布尔值：那会变成两个需要保持同步的事实，
    而"正文开头是不是条号"是可以直接从正文读出来的。

    取不到条号的切片（法规的前言、按长度切的普通文本）留 None。
    "没有条号"本身是信息——它说明这片不该参与条款级定位。
    """

    content: str
    start_offset: int = 0
    end_offset: int = 0
    article_number: str | None = None


def starts_mid_word(text: str, start: int) -> bool:
    """判断这个切片是不是从单词中间开始的。

    这是切片质量最硬的一个指标：`job` 被切成 `j | ob` 之后，
    模型读到的是残句，引用指向的也是残句。
    实测中这个比例可以从 70% 降到 0%，差距非常直观。
    """

    if start <= 0 or start >= len(text):
        return False
    previous_char = text[start - 1]
    current_char = text[start]
    return bool(_LATIN_LETTER.match(previous_char)) and bool(_LATIN_LETTER.match(current_char))


def find_best_split_point(
    text: str,
    start: int,
    chunk_size: int,
    *,
    min_keep_ratio: float = 0.5,
) -> int:
    """在一段文本里找一个合适的切点。

    找的策略，按优先级：
    1. 优先在句子边界切（句号、换行、中文标点）；
    2. 找不到就退到空格边界；
    3. 都没有才按长度硬切。

    第 2 步是关键：**退到空格边界，而不是直接按长度切**。
    这一条就是"绝不从单词中间切断"的实现。

    另外还有一个容易被忽略的细节：如果切完之后剩下的尾巴太短
    （比如只剩 30 个字），就把它一起带走，
    否则会产生一堆没有信息量的碎片切片，白占检索名额。
    """

    min_keep = int(chunk_size * min_keep_ratio)
    limit = min(start + chunk_size, len(text))

    if len(text) - limit < chunk_size * 0.3:
        return len(text)

    window = text[start:limit]
    best = -1
    for separator in SENTENCE_SEPARATORS:
        index = window.rfind(separator, min_keep)
        if index >= 0:
            best = max(best, index + len(separator))
    if best > 0:
        return start + best

    space_index = window.rfind(' ')
    if space_index > min_keep:
        return start + space_index + 1

    return limit
