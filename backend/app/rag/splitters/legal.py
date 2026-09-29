from __future__ import annotations

import re

from app.rag.splitters.base import (
    DEFAULT_CHUNK_SIZE,
    HARD_LIMIT_MULTIPLIER,
    SplitChunk,
    find_best_split_point,
)
from app.rag.splitters.semantic import split_semantic_text

"""法条感知切分。

为什么在"结构感知"之外还要单独做一种策略：

结构感知切分的原则是"在作者划好的边界切，但不跨边界合并"——可它给合并留了
一个口子：**只要总长不超过 500 字，相邻的小单元就会并成一片。**
在叙述型材料里这是对的（一段话本来就该连着上下文一起读）。

但在法规问答里这个口子会出问题：

    第28条 经营机构应当……（160 字）
    第29条 经营机构不得……（140 字）
    第30条 投资者应当……（150 字）

结构感知会把这三条并成一片 450 字的切片。后果有两个，都不轻：

1. **向量被"平均"掉了。** 一片包含三个不同主题的文本，
   它的向量是三个主题的折中。用户问第29条的事，
   这片向量和问题的相似度反而不如"只讲第29条"的那片——
   它要同时和另外两个主题竞争。
2. **引用说不清。** 答案依据是"《办法》第29条"，
   但切片里躺着三条。引用高亮时到底该高亮哪一段？

所以这里的规则只有一条，而且是硬规则：

    **一条 = 一个切片。绝不跨条合并。**

超过硬上限的长条才降级去切——但每一切片都会**带上条号前缀**，
这样即使一条被切成了三片，每一片单独看也仍然知道自己是谁。
（这一点很关键：切片脱离原文被塞进提示词时，
 没有条号的那片就是一截无法引用的野文。）

代价与取舍：
    会多出一些很短的切片（比如"第N条 本指引自发布之日起施行。"）。
    这是**故意接受的代价**——短切片只是偶尔占一个检索名额，
    而跨条合并会让引用失真，后者在合规场景里是硬伤。
"""

# "第…条"这个字样。用它来计数、打分、定位条文起点。
#
# 用"第…条"当切点，是因为它是法规里唯一稳定的原子单位：
# 编 / 章 / 节的跨度太大，款 / 项 的编号在不同文里写法不统一。
#
# 两种写法分开处理，原因是它们面对的排版噪声不一样：
#
#   1. 中文数字：**容忍被排版拆开**。
#      用"方正书版"排出来的证监会文件，抽出来的文本是一行一个片段：
#          …⏎第⏎十八⏎条⏎、⏎第十九条…⏎第⏎三⏎十九条⏎…
#      如果不容忍中间的换行，像"第⏎十八⏎条"这样的**真条文**会被整条漏掉。
#      实测《证券法》因此漏掉了 20 条（只能认出 206 / 226 条）。
#      容错窗口限制在 3 个字符以内：够容纳一个换行加缩进，
#      又不至于让两个不相干的东西被连起来。
#
#   2. 阿拉伯数字：**只认紧挨着的写法**。
#      因为页码也是阿拉伯数字。PDF 分页时页码会插进正文，形成
#          …第⏎⏎15⏎二十⏎六⏎条⏎…（真身是"第二十六条"，页码 15 插在了中间）
#      一旦允许阿拉伯数字跨空白匹配，就会把"15"当成条号的开头。
_NUM = '[一二三四五六七八九十百零〇]'
_ARTICLE_TOKEN = re.compile(
    rf'第(?:\s{{0,3}}{_NUM})+\s{{0,3}}条'   # 中文数字，容忍排版拆字
    rf'|第\d+条'                             # 阿拉伯数字，必须紧挨着
)

# 条号后面**不可能**紧接的字。
#
# 这一条是被真实数据打出来的。第一版只按"第…条"出现的位置切，
# 结果《证券期货投资者适当性管理办法》切出了 88 片，而这部办法只有 43 条——
# 多出来的一倍，全是这种句子造出来的：
#
#     （六）违反本办法第二十五条，未按规定录音录像……
#     （十）违反本办法第六条、第十八条至第二十四条……
#
# 这里的"第二十五条""第十八条"是**引用**，不是条文起点。
# 把引用当起点切，后果是把一条法规从中间劈开：
# 前半段（引用它的那条）和后半段（被引用的那条）各自少了半截。
#
# 区分办法很简单：真正的条文后面跟着正文，而引用后面跟的是连接词。
_CONNECTOR_AFTER = set('、，,。；;：:至和及与或称并且')

# 条号后面**不可能**紧接的引用续接词。
#
# 连接词只能挡住"第X条、""第X条，"这种一眼可见的枚举。
# 但引用还有一种写法，前后都没有标点：
#
#     第三十三条规定的，按照《证券投资基金法》……办理
#     本法第十二条第二款的规定
#     第八十五条第二款规定
#
# 这类引用靠标点判断不出来，只能认"条号后面紧跟的词"。
# 这里的取舍很明确：**只收条文绝不可能以之开头的词**。
# "第X条规定""第X条第N款""第X条第N项"——真条文不会这样开头，
# 所以命中即拒，不设例外。
_REFERENCE_AFTER = re.compile(
    r'^(?:规定|所称|所列|列明|要求|规定情形|'
    r'第[一二三四五六七八九十百零〇\d]+[款项])'
)

# 顺序过滤的尝试记录（已废弃，保留说明以免后人重走这条路）：
#
#   曾经加过一条"条号必须递增"的过滤——理由是法规条文按顺序编号，
#   往回跳的一定是引用。它对《期货和衍生品法》效果极好（155 条全部认对），
#   但在《证券法》上彻底失败：那份 PDF 的抽取顺序本身是乱的，
#   53~80 条出现在 81 条之后，递增过滤把它们全判成了引用，
#   226 条里直接误杀 76 条。更糟的是它还会把
#   "《期货交易管理条例》第六十七条"这种跨法规引用当成新条文收下
#   （67 比当时的 41 大），反而制造出假条界。
#
#   教训：**一个依赖"输入是干净的"的过滤，不能在输入不干净的地方用。**
#   位置判据对噪声是钝感的，顺序判据对噪声是敏感的——
#   而这里唯一确定的前提就是噪声一定存在。

# 条文之前**必须**出现的收尾符。或者说：条号不能凭空插在一句话中间。
_TERMINATOR_BEFORE = set('。；;：:\n')

# 段落开头的"短前缀"容忍度。
#
# 这一条来自《证券法》里一个很直观的排版事故：
#
#     第八章
#
#     证券公司第一百一十八条
#
#     设立证券公司，应当具备下列条件……
#
# 章标题（"证券公司"）和紧随其后的条号被抽到了同一行。
# 于是"第一百一十八条"前面是汉字"司"而不是断句符，
# 被上一条规则判成引用——整条法规凭空少了一条。
# 实测《证券法》因此丢了 5 条（118/145/164/168/224）、
# 《期货和衍生品法》丢了 2 条（118/125）。
#
# 判据是**段落开头 + 短前缀**，两条同时满足才放行：
#
#   - 段落开头（上一行是空行）：真条文总是新起一段；
#     而引用（"（六）本办法第二十九条规定的……"）永远在句子中间。
#   - 前缀短且不含标点：章标题就那么几个字。
#     加"不含标点"这一条，是为了避免把"……规定，（三）第X条"这种
#     列表项里的引用也放进来。
_PARAGRAPH_PREFIX_MAX = 12
_PREFIX_HAS_PUNCTUATION = re.compile(r'[。；：，、,;:]')

# 章标题行，例如"第八章"。
_CHAPTER_LINE = re.compile(r'^第[一二三四五六七八九十百零〇\d]{1,3}章')

# 判断"条号后面跟的是什么"时要跳过的空白。
#
# 这里**必须包含换行**，这一点踩过一次：
# 上面那种一行一个片段的 PDF，引用长这样——
#     …⏎第二十五条⏎经营机构通过营业网点…⏎本办法⏎第十二条⏎、⏎第二十条⏎、⏎…
# 「第十二条」后面紧跟的是换行，再才是顿号。如果判空时只跳过空格和制表符，
# 就会拿到一个换行符（不是顿号），于是把引用误判成条文起点——
# 《适当性管理办法》因此多切了 16 片，一部 43 条的办法切出了 59 条。
_WHITESPACE = ' \t\u3000\r\n'

# 判定"这像不像一份法规文本"的下限。
# 取 5 是因为：一份真正的法规不可能只有两三条；
# 而正文里偶然出现一两次"第一条"的问答类文件（比如证监会问答），
# 不应该被当成法条文本按条切。
LEGAL_MIN_ARTICLES = 5


def count_article_markers(text: str) -> int:
    """数一数这段文本里出现了多少次"第N条"。

    这是**宽松计数**：引用（"依据本办法第二十九条"）也算。
    它服务于两个不需要精确的场景——给解析容器打分、判断文档规模。
    需要精确定位条文时请用 article_boundaries()。
    """

    if not text:
        return 0
    return len(_ARTICLE_TOKEN.findall(text))


def find_article_tokens(text: str) -> list[str]:
    """宽松地找出文本里出现过的所有条号，按出现顺序返回。

    和 `article_boundaries` 的区别在于**用途不同**：

    - `article_boundaries` 是给切分用的，它要判断"哪里是一条的起点"，
      所以宁缺毋滥，用了一组严格的局部判据；
    - 这个函数是给**查询解析**用的。用户问"《证券期货投资者适当性管理办法》
      第二十九条"，这句话里"第二十九条"前面是"办法"而不是断句符，
      严格判据会把它挡掉——但用户的意思非常明确。

    所以查询侧的解析用宽松匹配，只找出"用户提到了哪些条号"，
    由更高层去判断这是不是一个引用。
    """

    if not text:
        return []
    return [re.sub(r'\s+', '', match.group(0)) for match in _ARTICLE_TOKEN.finditer(text)]


def starts_with_article(text: str) -> bool:
    """这一片正文是不是**以条号开头**。

    用途是区分"这一片就是第 X 条"和"这一片是第 X 条的续片"。
    两者都有 article_number，但只有前者是那一条的入口——
    精确检索时它们都要返回（用户要完整的一条），
    但加权和展示时不能一视同仁：把续片和条文头同等对待，
    会出现"一片没头没尾的续文排在了整条前面"。
    """

    return bool(_ARTICLE_TOKEN.match((text or '').strip()))


_CN_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_CN_UNITS = {'十': 10, '百': 100}


def chinese_number_to_int(token: str) -> int | None:
    """把"第二十九条"里的"二十九"转成 29。

    转换只用于**比较条号的先后**，不用于展示（展示一律用原文写法）。
    所以精度要求不高，但"第十条""第一百三十七条""第二百二十六条"
    这三种典型写法必须都对——它们分别代表了"缺省的一""进位"和"进位后再进位"。
    """

    body = re.sub(r'\s+', '', token).removeprefix('第').removesuffix('条')
    if not body:
        return None
    if body.isdigit():
        return int(body)

    total = 0
    section = 0
    digit = 0
    for char in body:
        if char in _CN_DIGITS:
            digit = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            # "第十条"里的"十"前面没有数字，按 1 算。
            section += (digit or 1) * _CN_UNITS[char]
            digit = 0
        else:
            return None
    return total + section + digit


def _article_candidates(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    """挑出"看起来像条文起点"的候选，并把拒掉的连同原因一起返回。

    判据有两条，必须同时满足：

    1. **前面是收尾符**（句号 / 分号 / 冒号 / 换行 / 全文开头）。
       条文之间必然有断句，所以真的条号不会插在一句话中间。
    2. **后面不是连接词**。真条文后面跟着实质内容；
       而"第二十九条，未按规定……"后面跟的是逗号，
       "第六条、第十八条"后面跟的是顿号——它们都是引用。

    这两条都是**位置判据**，对排版噪声不敏感：
    不需要去穷举"本办法""根据""违反"这些前缀词（那永远举不完），
    也不依赖文字的先后顺序（PDF 抽取顺序可能是乱的）。

    返回 (候选, 拒绝记录)。拒绝记录要往外传，因为"拒了什么"必须看得见——
    过滤逻辑一旦静默出错，表现是"某一条法规凭空消失了"，而这件事
    在成品里完全看不出来。
    """

    if not text:
        return [], []

    candidates: list[tuple[int, str]] = []
    rejected: list[tuple[int, str, str]] = []
    for match in _ARTICLE_TOKEN.finditer(text):
        token = re.sub(r'\s+', '', match.group(0))

        before = text[: match.start()].rstrip(' \t\u3000')
        if before and before[-1] not in _TERMINATOR_BEFORE and not _at_paragraph_start(before):
            rejected.append((match.start(), token, '条号前面不是断句符，判为引用'))
            continue

        after = text[match.end() :].lstrip(_WHITESPACE)
        if after and after[0] in _CONNECTOR_AFTER:
            rejected.append((match.start(), token, f'条号后面紧跟连接词"{after[0]}"，判为引用'))
            continue

        reference = _REFERENCE_AFTER.match(after)
        if reference:
            rejected.append(
                (match.start(), token, f'条号后面紧跟引用续接词"{reference.group(0)}"，判为引用')
            )
            continue

        candidates.append((match.start(), token))
    return candidates, rejected


def _at_paragraph_start(before: str) -> bool:
    """条号是否出现在一个"段落开头的短前缀"之后。

    用于容忍"章标题和条号被抽到同一行"这种排版事故，见 _PARAGRAPH_PREFIX_MAX 的说明。
    """

    line_start = before.rfind('\n') + 1
    prefix = before[line_start:].strip()
    if not prefix or len(prefix) > _PARAGRAPH_PREFIX_MAX:
        return False
    if _PREFIX_HAS_PUNCTUATION.search(prefix):
        return False
    # 上一行必须是空行（段落之间有空行），才认定这是新的一段。
    if not (line_start >= 2 and before[line_start - 2] == '\n'):
        return False

    # 再往上必须是章标题行。
    #
    # 最初只要求"段落开头的短前缀"，结果放进了一批假条界——
    # 《证券法》多出 13 个重号、《期货和衍生品法》多出 9 个，
    # 都是"段落开头恰好有几个字的引用"被误收进来。
    #
    # 要求上一行是"第X章"之后，这条规则就变得很具体了：
    # 它只处理"章标题后面的第一个条号被抽到章标题那一行"这一种情况，
    # 而这种情况恰恰是真实存在的排版事故。
    preceding = before[: line_start - 1].rstrip('\n')
    previous_line = preceding.rsplit('\n', 1)[-1].strip()
    return bool(_CHAPTER_LINE.match(previous_line))


def audit_article_boundaries(text: str) -> dict[str, list]:
    """返回条界判定的完整过程：接受了什么、拒绝了什么、为什么。

    为什么要专门做一个"审计"入口，而不是只看最终结果：

    过滤必然会误伤。误伤的后果是"两条法规被并进同一片"，
    而这件事在成品里看不出来——切片看起来正常，只是引用会指到隔壁那条。

    所以：**过滤必须留痕。** 体检报告会把被拒的候选连同原因一起打出来，
    人要能一眼扫过并判断"这里拒得对不对"。
    """

    accepted, rejected = _article_candidates(text)
    return {'accepted': accepted, 'rejected': rejected}


def article_boundaries(text: str) -> list[tuple[int, str]]:
    """找出真正的条文起点，返回 [(位置, 条号), ...]。

    判据全部是**局部**的：只看条号前后几个字，不看它在文档里的位置，
    也不看它和别的条号谁先谁后。

    这是刻意的。局部判据对排版噪声不敏感——不管 PDF 把文字排成什么样、
    页序是不是乱的、页码有没有插进正文，只要"第X条"前后是干净的，
    它就是条文起点。这一点在《证券法》上验证过：
    那份 PDF 的抽取顺序本身就是乱的（53~80 条排在 81 条后面），
    任何依赖顺序的判据都会在这里翻车。
    """

    return audit_article_boundaries(text)['accepted']


def looks_legal(text: str, *, min_articles: int = LEGAL_MIN_ARTICLES) -> bool:
    """判断这像不像法规文本。

    用**严格**的条文数量判断，不用宽松计数。

    这个区别是被真实数据逼出来的：《证券期货投资者适当性管理办法》问答
    里有 6 处"第N条"，但全部是引用（"应符合第八条规定"），
    它是一份问答，不是一部法规。用宽松计数会把它错判成法规，
    然后按条切——切出来的每一片都不成句。
    """

    return len(article_boundaries(text)) >= min_articles


def _article_spans(text: str) -> list[tuple[int, int, str]]:
    """切出每一"条"的区间，返回 (起点, 终点, 条号)。

    条号从匹配到的原始文本里取（"第二十九条"），不自己换算成阿拉伯数字——
    换算规则在"第〇条""第一百一十条"这类写法上容易出错，
    而原文里的写法本来就够用，改不得。
    """

    matches = article_boundaries(text)
    spans: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        start, article_no = match
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        spans.append((start, end, article_no))
    return spans


def split_legal_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    hard_limit_multiplier: float = HARD_LIMIT_MULTIPLIER,
) -> list[SplitChunk]:
    """按"条"切分法规文本。"""

    if not text:
        return []

    spans = _article_spans(text)
    if not spans:
        # 一个条界都没有，按条切无从下手。交给结构感知——
        # 它至少还会认标题和句子边界。
        return split_semantic_text(text, chunk_size=chunk_size)

    # ⚠️ 这里曾经写成 "len(spans) < 2 就退回按结构切"，那是个错误。
    #
    # 原因在于切分的输入是**section**，而不是整篇文档；而 PDF 的 section 是"一页"。
    # 一页里常常只放得下一条法规的开头（或者上一条的结尾加下一条的开头），
    # 于是这一页只能识别出 1 个条界。旧写法在这种情况下会退回按句子边界切，
    # 结果就是**那个条界被丢掉了**——一条完整的法规被并进了上一片。
    #
    # 实测表现：《证券期货投资者适当性管理办法》第二十九条出现在
    # "…予以明确。 第二十九条 经营机构应当…" 这一片里，片首还带着上一页的页码，
    # 而引用它的时候只会显示"第二十八条"。
    #
    # 1 个条界也要用它：把这一页切成"前半段 + 条文"，
    # 至少保证"第 X 条从这里开始"这件事不被丢掉。

    hard_limit = int(chunk_size * hard_limit_multiplier)
    # 三元组多出来的那一项就是"这一片属于哪一条"。
    # 续片和超长条切出来的每一小片，都带着**同一条**的条号——
    # 这正是条款级定位能成立的前提。
    ranges: list[tuple[int, int, str | None]] = []

    # 第一条之前的内容（文号、发布日期、通过会议等）单独成片。
    # 它承载的是**元数据**而不是规则，扔掉会让"这份文件什么时候生效的"无处可查。
    preamble_end = spans[0][0]
    if preamble_end > 0 and text[:preamble_end].strip():
        # 前言没有条号。留 None 而不是硬塞一个，是因为"没有条号"这件事
        # 会影响条款级定位——它不该被当成任何一条法规。
        ranges.append((0, preamble_end, None))

    for start, end, article_no in spans:
        if end - start <= hard_limit:
            ranges.append((start, end, article_no))
            continue

        # 超长条：在条内按句子边界切。
        # 注意这里不去动切片内容——条号本来就躺在原文里（每条的文本都以它开头），
        # 从原文切片自然就带着它，不需要额外拼接。拼接反而会改变原文。
        position = start
        while (end - position) > hard_limit:
            cut = find_best_split_point(text, position, chunk_size)
            cut = min(cut, end)
            if cut <= position:
                cut = min(position + chunk_size, end)
            ranges.append((position, cut, article_no))
            position = cut
        if position < end:
            ranges.append((position, end, article_no))

    # 注意：这里**没有** _absorb_tiny_ranges。
    # 结构感知切分用它把碎片并进邻居，但在这个策略下，
    # 把"第N条"并进"第N-1条"恰恰是我们要避免的事。
    # 宁可留一个短切片，也不要让两条法规共享一片。
    chunks: list[SplitChunk] = []
    for start, end, article_number in ranges:
        content = text[start:end]
        if not content.strip():
            continue
        chunks.append(
            SplitChunk(
                content=content,
                start_offset=start,
                end_offset=end,
                article_number=article_number,
            )
        )
    return chunks


__all__ = [
    'LEGAL_MIN_ARTICLES',
    'audit_article_boundaries',
    'article_boundaries',
    'chinese_number_to_int',
    'count_article_markers',
    'find_article_tokens',
    'looks_legal',
    'starts_with_article',
    'split_legal_text',
]
