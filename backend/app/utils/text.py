from __future__ import annotations

import re

_MULTI_BLANK_LINES = re.compile(r'\n{3,}')
# 英文 PDF 里很常见的排版断词：行尾是连字符，下一行接小写字母，例如 "independ-\nence"
_HYPHEN_LINE_BREAK = re.compile(r'(\w)-\n(\w)')

# 中文字符之间被排版工具插进去的空格。
#
# 这是一个**只在中日韩文档里才会出现**的问题，但它对检索的破坏很大：
# 用方正书版排出来的证监会文件，抽出来的文本长这样——
#
#     为了规范证券期货 投资者 适当性管理 ， 维护 投资者合法权益
#
# 空格来自排版时的字块边界，不是原文的一部分。中文本来就不用空格分词，
# 所以这些空格必须去掉。留着的话有两个后果：
#
#   1. BM25 的分词会被切错。jieba 遇到 "适当性管理 ，" 这种带空格的串，
#      切出来的 token 和没有空格时不一样，同一个意思会变成两组不同的词——
#      索引时是一组、查询时是另一组，于是**检索直接失效**。
#   2. 切片正文展示给用户时是断开的，引用看起来像排版坏掉的文件。
#
# 只处理"两侧都是中文/全角字符"的空格（含制表符和全角空格）。
# 为什么限定两侧都是中文：中文和英文/数字之间的空格是**有意义**的
# （"第 3 条"、"Basel III 的规定"），去掉会把它们粘成一团。
_CJK = r'\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
_CJK_INNER_SPACE = re.compile(rf'(?<=[{_CJK}])[ \t]+(?=[{_CJK}])')

# 条号 / 章号后面的分隔空格。
#
# 上面那条规则会把"第二十九条 经营机构……"里的空格一并删掉，变成
# "第二十九条经营机构……"。语义没错，但引用展示时读起来挤在一起。
#
# 所以先把它换成**全角空格**：正式法规文本本来就用全角空格做这个分隔，
# 而且全角空格不会被上面的规则再删一次（那条规则只删半角空格和制表符）。
# 这样既保住了检索要的连续性，也保住了阅读上的分隔感。
# （不用后行断言：Python 的 re 不支持变长后行断言，
#   而"第X条"的长度是可变的，所以改用捕获组。）
_ARTICLE_HEADING_SPACE = re.compile(r'(第[一二三四五六七八九十百零〇\d]{1,4}[条章节编])[ \t]+')

# 超长行往往是"双栏被串成一行"的信号，用来做解析质检
LONG_LINE_THRESHOLD = 300

# ---- 还原断行（只用于 PDF）----
#
# 问题的样子：用方正书版排出来的法规 PDF，抽文本时**每个文字片段占一行**：
#
#     第一条
#     为了规范证券期货
#     投资者
#     适当性管理
#     ，
#     维护
#     投资者合法权益
#
# 这些换行不是原文的段落，是排版时文本框边界的产物。
#
# 为什么必须处理它（这一条被真实检索结果打出来过）：
#
#   1. **BM25 的中文分词会在换行处断开。** 分词器的中文片段规则是
#      `[\u4e00-\u9fff]+`，它不跨换行。于是"投资者"和"适当性管理"
#      被切成两段独立的分词单元，"投资者适当性管理"这个短语检索不到。
#   2. **向量质量下降。** 送进 embedding 的是一堆被换行切碎的短语，
#      而不是完整句子。
#   3. 引用展示给用户看的时候，是断开的。
#
# 怎么判断该不该把两行接起来——三条规则，缺一不可：
#
#   a. 上一行**没有**以句末标点结尾（。！？；：）。
#      以句末标点结尾 = 一个语义单元结束了，换行是真的。
#   b. 下一行**不是**结构标记。
#      这一条是关键：它保住了"第一条""第一章"前面的换行。
#      丢掉它的话，法规名会和"第一条"粘成一句，
#      而条界判定要求条号前面是断句符——第一条会被整条漏掉。
#   c. 上一行不是纯页码。
#      PDF 分页时页码会插进正文流，把它和标题粘起来只会制造噪声。
_SENTENCE_END = '。！？；：'

_STRUCTURE_START = re.compile(
    r'^(?:'
    r'第[一二三四五六七八九十百零〇\d]{1,4}[条章节编]'   # 第X条 / 第X章
    r'|[一二三四五六七八九十]+、'                        # 一、
    r'|（[一二三四五六七八九十]+）'                       # （一）
    r'|\([一二三四五六七八九十]+\)'                       # (一)
    r'|#{1,6}\s'                                         # Markdown 标题
    r'|[IVX]{1,5}\.\s'                                   # 罗马数字小标题
    r')'
)

# 纯页码行，例如 "1"、"12"、"— 9 —"
_PAGE_NUMBER_ONLY = re.compile(r'^[\s—\-–]*\d{1,4}[\s—\-–]*$')

# 上一行结尾是孤零零的"第"，说明条号自己也被排版拆开了——
# 原文是"第一百一十八条"，被排成"第" + "一百一十八条"两行。
#
# 这种行必须**保持断开**，不能和上一行接起来。原因：
# 条界判定要求"第X条"前面是断句符；一旦把"第"粘到上一句话的末尾，
# 合并后的条号前面就变成了普通汉字，这一条会被整条漏掉。
#
# 实测：《证券法》因此丢了 5 条（118/145/164/168/224），
# 《期货和衍生品法》丢了 2 条（118/125）。
_TRAILING_DI = re.compile(r'第$')

_ENDS_WITH_CJK = re.compile(rf'[{_CJK}]$')
_STARTS_WITH_CJK = re.compile(rf'^[{_CJK}]')


def unwrap_pdf_lines(text: str) -> str:
    """把 PDF 里"一行一个片段"的换行还原成正常段落。

    §只用于 PDF。§

    docx 和 HTML 的换行是**作者写的真实段落结构**，
    对它们做同样的还原会把相邻段落错误地粘成一段。
    PDF 的换行则大多是排版产物——这是格式本身的差别，不是数据好坏的差别。
    """

    if not text:
        return ''

    lines = text.split('\n')
    merged: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append('')          # 空行 = 段落边界，保留
            continue

        if merged and merged[-1]:
            previous = merged[-1]
            can_join = (
                previous[-1] not in _SENTENCE_END
                and not _STRUCTURE_START.match(stripped)
                and not _PAGE_NUMBER_ONLY.match(previous)
                and not _TRAILING_DI.search(previous)
            )
            if can_join:
                # 中文之间直接接上；拉丁文字之间补一个空格，
                # 否则英文 PDF 会被粘成 "thequickbrownfox"。
                separator = (
                    '' if _ENDS_WITH_CJK.search(previous) or _STARTS_WITH_CJK.match(stripped)
                    else ' '
                )
                merged[-1] = previous + separator + stripped
                continue

        merged.append(stripped)

    return '\n'.join(merged)


def clean_text(text: str) -> str:
    """基础清洗。

    做五件事，都是实测中确实会污染检索的：
    1. 统一换行符（Windows 的 \\r\\n 会让后面的切分逻辑行为不一致）；
    2. 修掉行尾连字符断词；
    3. **去掉中文字符之间被排版插进去的空格**（见 _CJK_INNER_SPACE）；
    4. 去掉行尾多余空格；
    5. 把连续空行压缩成空行。

    刻意不做的事：**不动正文里的标点和大小写**。
    清洗的目标是去掉"排版噪音"，不是改写内容——
    改写内容会让引用指向的原文和用户看到的对不上。
    """

    if not text:
        return ''

    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized = _HYPHEN_LINE_BREAK.sub(r'\1\2', normalized)
    # 替换串里不能写 \u3000（re 不认这个转义），所以用真正的全角空格字符。
    normalized = _ARTICLE_HEADING_SPACE.sub('\\1\u3000', normalized)
    normalized = _CJK_INNER_SPACE.sub('', normalized)
    normalized = '\n'.join(line.rstrip() for line in normalized.split('\n'))
    normalized = _MULTI_BLANK_LINES.sub('\n\n', normalized)
    return normalized.strip()


def count_long_lines(text: str, *, threshold: int = LONG_LINE_THRESHOLD) -> int:
    """统计超长行数量。

    用途是解析质检：双栏排版的 PDF 被按坐标顺序抽文本时，
    左右两栏会串成一行，表现为"一行特别长"。
    这个指标不完美，但足以在入库前提示"这份文件的解析可能有问题"。
    """

    return sum(1 for line in text.split('\n') if len(line) > threshold)


def estimate_token_count(text: str) -> int:
    """粗略估算 token 数。

    只用来说明"这段文本大概有多长"，不用于计费。
    做法是英文按空格、中文按字符估一个折中值——
    真实的 token 计算需要模型的分词器，这里不值得引入那个依赖。
    """

    if not text:
        return 0
    return max(1, int(len(text) / 3))
