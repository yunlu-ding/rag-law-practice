from __future__ import annotations

import hashlib
import re
from datetime import date

"""法规元数据抽取。

这份模块回答一个问题：**这份文件到底是什么、由谁发的、现在还有效吗？**

为什么元数据值得单独一个模块：

纯文本 RAG 只关心"哪几段话像答案"。但法规问答里，
用户真正要的不只是"有一句话说 X"，而是：

    "《证券期货投资者适当性管理办法》第二十九条（部门规章，现行有效，2020-12-01 施行）"
    —— 这段话的作者是谁、算不算数、什么时候开始算数。

同一条规则，写在《证券法》里和写在协会自律规则里，**效力完全不同**。
模型不告诉用户这件事，用户就会把自律规则的软要求当成法定义务。
所以元数据不是装饰，它是答案的一部分。

设计上的一条硬原则：**每个抽出来的字段都带 evidence（出处）。**

抽取必然是"猜"——正则猜不中"施行"和"公布"的区别是常有的事。
但如果每条结论都能说清"我是从哪句话里看出来的"，
人工核对就只是核对，不需要重新读一遍全文。
反过来，如果只给一个孤零零的日期，人只能选择相信或不信。
"""

# ---------------------------------------------------------------------------
# 法的层级
#
# 效力从高到低：法律 > 行政法规 > 部门规章 > 规范性文件 > 自律规则。
# rank 就是为这个顺序服务的——冲突时以上位法为准，这句话必须能被代码执行，
# 而不能只写在文档里。
# ---------------------------------------------------------------------------

LEGAL_LEVELS: dict[str, dict[str, object]] = {
    'law': {'label': '法律', 'rank': 1},
    'admin_regulation': {'label': '行政法规', 'rank': 2},
    'department_rule': {'label': '部门规章', 'rank': 3},
    'normative_document': {'label': '规范性文件', 'rank': 4},
    'self_regulation': {'label': '自律规则', 'rank': 5},
}

# 语料目录名 → 层级。层级来自目录，因为目录是**人工确认过**的，
# 比让正则去猜发布主体准得多。这一步是刻意选择"信人"而不是"信算法"。
LEGAL_LEVEL_BY_FOLDER = {
    '法律层级': 'law',
    '行政法规': 'admin_regulation',
    '部门规章': 'department_rule',
    '规范性文件': 'normative_document',
    '自律规则': 'self_regulation',
}

VALIDITY_LABELS = {
    'effective': '现行有效',
    'draft': '征求意见稿',
    'superseded': '已被修订',
    'repealed': '已废止',
}


def level_rank(legal_level: str | None) -> int | None:
    if not legal_level:
        return None
    info = LEGAL_LEVELS.get(legal_level)
    return int(info['rank']) if info else None


def level_label(legal_level: str | None) -> str:
    if not legal_level:
        return '未分类'
    info = LEGAL_LEVELS.get(legal_level)
    return str(info['label']) if info else legal_level


# ---------------------------------------------------------------------------
# 发布机关
#
# 按"从具体到宽泛"排列。之所以要有顺序：一份证监会令的正文里
# 会同时出现"国务院"和"中国证监会"，先匹配谁决定了结果对不对。
# ---------------------------------------------------------------------------

REGULATOR_PATTERNS: dict[str, re.Pattern[str]] = {
    '中国证券监督管理委员会': re.compile(r'中国证券监督管理委员会|证监会'),
    '上海证券交易所': re.compile(r'上海证券交易所|上交所'),
    '深圳证券交易所': re.compile(r'深圳证券交易所|深交所'),
    '北京证券交易所': re.compile(r'北京证券交易所|北交所'),
    '中国证券业协会': re.compile(r'中国证券业协会'),
    '中国证券投资基金业协会': re.compile(r'中国证券投资基金业协会'),
    '中国期货业协会': re.compile(r'中国期货业协会'),
    '全国人民代表大会常务委员会': re.compile(r'全国人民代表大会|全国人大常委会'),
    '国务院': re.compile(r'国务院'),
}

# 层级 → 可能的发布机关（按可能性排序）。
#
# 这一步是"用已经确认的字段去约束还没确认的字段"。
#
# 法律层级来自语料目录，是**人工确认过**的；而发布机关是从正文里正则搜出来的。
# 让后者的取值范围被前者限住，能挡掉一大类错误：
# 《证券公司监督管理条例》全文里"中国证券业协会"出现过（某一条提到它），
# 而"中国证券监督管理委员会"一次都没出现（条例的写法是"国务院证券监督管理机构"）——
# 于是纯全文搜索会把发布机关判成中证协，而正确答案是国务院。
#
# 已知的代价：如果某份文件被放进了错误的层级目录，这里会跟着错。
# 但层级目录本来就是人工核对过的，这个前提比"全文里谁先出现"可靠得多。
REGULATORS_BY_LEVEL: dict[str, tuple[str, ...]] = {
    'law': ('全国人民代表大会常务委员会', '国务院'),
    'admin_regulation': ('国务院',),
    'department_rule': ('中国证券监督管理委员会', '国务院'),
    'normative_document': (
        '中国证券监督管理委员会', '上海证券交易所', '深圳证券交易所',
        '北京证券交易所', '国务院',
    ),
    'self_regulation': (
        '上海证券交易所', '深圳证券交易所', '北京证券交易所',
        '中国证券业协会', '中国证券投资基金业协会', '中国期货业协会',
        '中国证券监督管理委员会',
    ),
}

# 发布机关一定写在文件抬头，不会藏在正文深处。
# 所以先在开头找；找不到才扩大范围，并且把"是在全文里找到的"记进 evidence。
_REGULATOR_HEAD_CHARS = 800

# 文号。三种常见写法，实测都出现过：
#   〔2020〕130号        部委公告类
#   第 XX 号令           主席令 / 国务院令
#   证监会令第 XX 号     部门规章
_DOC_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'[〔\[［【]\s*(?:19|20)\d{2}\s*[〕\]］】]\s*第?\s*\d+\s*号'),
    re.compile(r'中国证券监督管理委员会令\s*第\s*[一二三四五六七八九十百零〇\d]+\s*号'),
    re.compile(r'第\s*[一二三四五六七八九十百零〇\d]+\s*号令'),
]

_DATE = re.compile(r'((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日')

# 文件名里的日期后缀，例如 中华人民共和国证券法_20191228.pdf
_FILENAME_DATE = re.compile(r'_((?:19|20)\d{2})(\d{2})(\d{2})$')

# 日期前的语境词。窗口取 12 个字——实测"自2020年12月1日起施行"
# 这类表述里，关键词和日期的距离很少超过 10 个字。
#
# ⚠️ 窗口必须**前后都看**，这一点是被真实数据纠出来的。
# 中文法规里"公布""施行"这两个词，写在日期前面和后面都很常见：
#     经 2017 年 3 月 1 日国务院第 X 次常务会议通过   ← 关键词在前
#     自 2017 年 7 月 1 日起施行                      ← 关键词在后
# 最早只往回看，结果"自2017年7月1日起施行"这句话里的日期
# 一个都没被认出来，反而是正文深处某句顺带提及的日期被当成了施行日。
_DATE_CONTEXT_WINDOW = 12
_ISSUED_HINTS = ('公布', '发布', '通过', '修订', '修正', '批准')
_EFFECTIVE_HINTS = ('施行', '实施', '生效', '执行')

# 适用范围的关键词表。
# 用固定词表而不是自由抽取，是因为"范围"要用来做**检索过滤**——
# 过滤条件必须可比、可枚举，自由文本没法比。
SCOPE_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ('证券', re.compile(r'证券')),
    ('期货', re.compile(r'期货')),
    ('基金', re.compile(r'基金')),
    ('衍生品', re.compile(r'衍生品|衍生工具')),
]

# 征求意见稿的识别。这类文件**不是生效依据**，
# 混进知识库会让系统拿"还没定的规则"回答"现在该怎么做"。
_DRAFT_MARKERS = ('征求意见稿', '征求意见', '草案')

# 未发布标准封面的占位符。
#
# 这一条来自一个具体的排查。语料里有一份中证协团体标准
# 《证券公司投资者适当性管理规范》，封面上写着：
#
#     T/SAC 00X—2026
#     2026-**-** 发布    2026-**-** 实施
#
# 全文一次都没出现"征求意见"这几个字，所以按关键词判断它会通过。
# 但这两行本身就是结论：**标准号还没分配、发布日期是个占位符**，
# 说明它还没正式发布。
#
# 判据因此改为"看它有没有还没填上的占位符"——
# 一份正式发布的标准，不可能连自己的编号和日期都不知道。
_PLACEHOLDER_MARKERS = (
    re.compile(r'T/[A-Z]+\s*0*X'),   # 标准号占位：T/SAC 00X
    re.compile(r'\*{2,}'),           # 日期占位：2026-**-**
)


def _normalize_date(year: str, month: str, day: str) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _extract_dates(text: str) -> dict[str, list[tuple[str, str]]]:
    """按语境把日期分到公布/施行两栏，并记下依据。

    返回 {'issued': [(日期, 依据), ...], 'effective': [...]}。
    同一句话可能同时命中，所以两栏都要收集，最后由调用方按优先级决定。
    """

    buckets: dict[str, list[tuple[str, str]]] = {'issued': [], 'effective': []}

    for match in _DATE.finditer(text):
        normalized = _normalize_date(match.group(1), match.group(2), match.group(3))
        if not normalized:
            continue
        window_start = max(0, match.start() - _DATE_CONTEXT_WINDOW)
        window_end = min(len(text), match.end() + _DATE_CONTEXT_WINDOW)
        window = text[window_start : window_end]
        evidence = f'…{" ".join(window.split())}…'

        if any(hint in window for hint in _EFFECTIVE_HINTS):
            buckets['effective'].append((normalized, evidence))
        if any(hint in window for hint in _ISSUED_HINTS):
            buckets['issued'].append((normalized, evidence))

    return buckets


def _compact(text: str) -> str:
    """去掉所有空白。

    用途只有一个：匹配发布机关名称。

    为什么必须这样：用方正书版排出来的团体标准，封面上的机构名是一个字一个
    文本框摆出来的，抽出来长这样——

        中 国 证 券 业 协 会 发布

    直接拿"中国证券业协会"去正则匹配，一个字都对不上。
    去掉空白之后才是一个整体。

    这个操作对**机构名匹配**是安全的：机构名里本来就不含空白。
    但它不适合用来做别的判断（比如找日期语境），
    因为去掉空白会让不相干的字粘在一起。
    """

    return re.sub(r'\s+', '', text)


def _clean_filename_title(filename: str) -> str:
    """从文件名推出标题：去掉扩展名、日期后缀、下载序号。

    只去这些东西，**不动标题本身**——包括《》和（试行）。
    它们是标题的一部分，去掉了反而不准确。
    """

    stem = filename.rsplit('.', 1)[0]
    stem = re.sub(r'_\d{6,8}$', '', stem)
    stem = re.sub(r'\s*\(\d+\)$', '', stem)
    return stem.strip()


def _looks_like_title_line(line: str) -> bool:
    """判断一行像不像标题。"""

    if not line or len(line) > 60:
        return False
    if _DATE.search(line):
        return False
    # 抬头类词（主席令 / 国务院令 / ICS / 页码 / 目次）都不是标题
    if line.startswith(('（', '(', 'ICS', 'CCS', '第', '—', '-', '附件')):
        return False
    if line in ('目次', '目录', '前言', '总则'):
        return False
    # 必须含有中文，且不是一串编号
    if not re.search(r'[\u4e00-\u9fff]', line):
        return False
    return not re.fullmatch(r'[\d\s.、]+', line)


def _extract_title(text: str, filename: str) -> tuple[str, str]:
    """标题以**文件名**为准，正文首行只在文件名没有信息量时才用。

    这个优先级是实测之后反过来定的。最初的做法是"正文首行优先"，
    理由是文件名常带日期和下载序号——但那是可以用正则去掉的，
    而正文首行的问题没法用正则解决：

        中华人民共和国证券法.pdf        → 首行是"中华人民共和国主席令"
        证券交易业务指南第1号.pdf        → 首行是"— 1 —"（页码）
        证券公司投资者适当性管理规范.pdf   → 首行是"ICS 03.060"（标准封面）
        公开募集…适当性管理细则.pdf       → 首行是"1"

    这几份文件的名字本身就是人整理好的、准确的法规名。
    **优先信人整理过的信息，而不是信正则能从版式噪声里猜出什么。**

    正文首行只在文件名没有信息量时启用——典型场景是用户直接上传
    一个扫描件，文件名是 `scan001.pdf` 或者 `未命名.pdf`。
    """

    stem = _clean_filename_title(filename)
    # "有信息量"的判据：含中文、且长度够。像 "1"、"scan001"、"未命名" 这种才走正文。
    if len(stem) >= 6 and re.search(r'[\u4e00-\u9fff]', stem):
        return stem, f'文件名（人工命名）：{stem}'

    for line in text.split('\n')[:20]:
        candidate = line.strip()
        if _looks_like_title_line(candidate):
            return candidate, f'正文首部（文件名信息量不足）：{candidate[:30]}'

    return stem, '文件名（正文里也没有可用的标题行）'


def _extract_scope(text: str) -> tuple[list[str], str]:
    """从总则里认适用范围。

    只看正文前 2,000 字：适用范围条款按惯例写在总则，
    而正文后段出现"基金"很可能只是某一条的例子，不代表整部法适用基金。
    这个限制是刻意的——**宁可漏标，不要错标**，
    错标会让检索过滤把该看到的文件挡在外面。
    """

    head = text[:2000]
    found = [label for label, pattern in SCOPE_KEYWORDS if pattern.search(head)]
    if not found:
        return [], '未在前 2000 字里识别到适用范围关键词'
    return found, f'总则关键词命中：{"、".join(found)}'


def _extract_validity(text: str, title: str) -> tuple[str, str]:
    head = title + '\n' + text[:1500]
    for marker in _DRAFT_MARKERS:
        if marker in head:
            return 'draft', f'标题/开头出现"{marker}"'
    for pattern in _PLACEHOLDER_MARKERS:
        match = pattern.search(head)
        if match:
            return 'draft', f'开头出现未填充的占位符"{match.group(0)}"，说明尚未正式发布'
    return 'effective', '未发现征求意见或废止标记，默认视为现行有效（需人工复核）'


def extract_metadata(
    *,
    filename: str,
    text: str,
    legal_level: str | None = None,
) -> dict[str, object]:
    """从正文里抽取法规元数据。

    返回的每个业务字段都配一条 evidence；抽不到的字段返回 None / 空列表，
    并且一样给出 evidence 说明"为什么没抽到"——
    **"没抽到"和"没有这个信息"是两件事**，前者要人工补。

    `legal_level` 为空时（网页上传的场景，用户没选层级）会尝试推断，
    并在 evidence 里标明"这是系统建议"。
    推断**只在没给值时发生**——批量入库时层级来自人工核对过的目录，
    那种情况下不能被推断覆盖。
    """

    evidence: dict[str, str] = {}

    title, evidence['title'] = _extract_title(text, filename)

    # 发布机关：先按层级缩小候选范围，再先抬头后全文地找。
    #
    # 两层约束叠加的理由见 REGULATORS_BY_LEVEL 的说明：
    # 单靠"全文里谁出现过"会被正文里的顺带提及带偏。
    regulator: str | None = None
    candidates = REGULATORS_BY_LEVEL.get(legal_level or '', tuple(REGULATOR_PATTERNS))
    # 匹配在"去空白"的文本上做，理由见 _compact。
    for scope_name, scope_text in (
        ('抬头', _compact(text[:_REGULATOR_HEAD_CHARS])),
        ('正文', _compact(text)),
    ):
        for label in candidates:
            pattern = REGULATOR_PATTERNS.get(label)
            if pattern is None:
                continue
            match = pattern.search(scope_text)
            if match:
                regulator = label
                evidence['regulator'] = (
                    f'{scope_name}出现"{match.group(0)}"（去空白后第 {match.start()} 字符处）'
                )
                break
        if regulator:
            break
    if regulator is None:
        evidence['regulator'] = '正文里没有识别到已知发布机关名称'

    doc_number: str | None = None
    compact_text = _compact(text)
    for pattern in _DOC_NUMBER_PATTERNS:
        match = pattern.search(compact_text)
        if match:
            doc_number = match.group(0).strip()
            evidence['doc_number'] = f'正文出现文号"{doc_number}"（去空白后匹配）'
            break
    if doc_number is None:
        evidence['doc_number'] = '正文里没有识别到文号'

    dates = _extract_dates(text)

    issued_date: str | None = None
    if dates['issued']:
        issued_date, evidence['issued_date'] = dates['issued'][0]
    else:
        filename_match = _FILENAME_DATE.search(filename.rsplit('.', 1)[0])
        if filename_match:
            issued_date = _normalize_date(
                filename_match.group(1), filename_match.group(2), filename_match.group(3)
            )
            evidence['issued_date'] = f'文件名日期后缀（正文里没有公布日期）'
        else:
            evidence['issued_date'] = '正文与文件名里都没有公布日期'

    effective_date: str | None = None
    # 施行日的取法，优先级从高到低：
    #
    # ① **文件抬头的括注**。很多规章在标题下直接写一行摘要：
    #       （自2017年7月1日起施行，2024年6月修订）
    #    这一行是发布方自己写的摘要，权威性最高，也最容易读。
    # ② **最后一处带"施行"语境的日期**。正文里会顺带提到很多历史日期
    #    （"自2012年9月27日起施行"出现在某一条的说明里），
    #    而真正定调的那句"本办法自X年X月X日起施行"写在附则，也就是最后一条。
    #    第一处往往是噪声，最后一处更接近结论。
    #
    # ⚠️ 这两条都是**启发式**，不是保证。实测《期货交易管理条例》仍然取错
    #    （取到了被它取代的暂行条例的施行日）。这正是 evidence 要存在的原因：
    #    抽错的结论旁边就摆着它是从哪句话里来的，人一眼能看出问题。
    head_effective = _extract_dates(text[:300])['effective']
    if head_effective:
        effective_date, detail = head_effective[0]
        evidence['effective_date'] = f'文件抬头括注：{detail}'
    elif dates['effective']:
        effective_date, evidence['effective_date'] = dates['effective'][-1]
    else:
        evidence['effective_date'] = '正文里没有"施行/生效"类的日期'

    validity, evidence['validity'] = _extract_validity(text, title)

    scope, evidence['scope'] = _extract_scope(text)

    if not legal_level:
        legal_level, evidence['legal_level'] = _infer_legal_level(
            title=title,
            regulator=regulator,
            doc_number=doc_number,
            text=text,
        )
    else:
        evidence['legal_level'] = f'由语料目录确定（人工核对过）：{level_label(legal_level)}'

    return {
        'title': title,
        'legal_level': legal_level,
        'level_rank': level_rank(legal_level),
        'regulator': regulator,
        'doc_number': doc_number,
        'scope': scope,
        'issued_date': issued_date,
        'effective_date': effective_date,
        # 只留现行有效版本的前提下，这份文件本身没有"到期日"这个概念。
        # 字段保留是为了将来接版本链时不用改表结构。
        'expiry_date': None,
        'validity': validity,
        'content_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'evidence': evidence,
    }


# 发布机关 → 层级。只在用户没指定层级时用作兜底。
_LEVEL_BY_REGULATOR = {
    '全国人民代表大会常务委员会': 'law',
    '国务院': 'admin_regulation',
    '中国证券监督管理委员会': 'department_rule',
    '上海证券交易所': 'self_regulation',
    '深圳证券交易所': 'self_regulation',
    '北京证券交易所': 'self_regulation',
    '中国证券业协会': 'self_regulation',
    '中国证券投资基金业协会': 'self_regulation',
    '中国期货业协会': 'self_regulation',
}


def _infer_legal_level(
    *,
    title: str,
    regulator: str | None,
    doc_number: str | None,
    text: str,
) -> tuple[str | None, str]:
    """推断法的层级。

    这个函数服务的场景是"用户直接拖一个文件进网页"——那时没人告诉系统
    这是什么层级。批量入库不走这里（层级来自人工核对过的目录）。

    ⚠️ 它只能给建议，不能当结论。原因是证监会既发部门规章（"令"）
    也发规范性文件（"公告"），两者发布机关完全相同，只能靠公文形式区分：

        中国证券监督管理委员会令 第XXX号   → 部门规章
        中国证券监督管理委员会公告〔2020〕XX号 → 规范性文件

    而"公告"和"令"在文件名里经常看不出来，得读正文。
    """

    level = _LEVEL_BY_REGULATOR.get(regulator or '')
    if level is None:
        return None, '发布机关未识别，无法推断层级，需要人工指定'

    if level == 'department_rule':
        # 证监会发的文件，是"令"还是"公告"，决定了它是部门规章还是规范性文件。
        # 这个区分不是抠字眼：部门规章可以直接作为处罚依据，规范性文件不行。
        head = doc_number or title + text[:300]
        if '令' in head:
            return 'department_rule', f'发布机关={regulator} 且文号为"令"，推断为部门规章（需人工确认）'
        return 'normative_document', (
            f'发布机关={regulator} 但未见"令"字号，推断为规范性文件（需人工确认）'
        )

    return level, f'由发布机关"{regulator}"推断（需人工确认）'


__all__ = [
    'LEGAL_LEVEL_BY_FOLDER',
    'LEGAL_LEVELS',
    'VALIDITY_LABELS',
    'extract_metadata',
    'level_label',
    'level_rank',
]
