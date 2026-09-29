"""一次性实验：词条路由的几种打分方式，谁选得准。

起因是一个**很难解释的现象**：语义路由上线后，路由命中数从 10 涨到 13，
但评测通过率一点没动（81% / 60%）。用 `工具/诊断词条路由.py` 摊开看，
根因是**路由选错了词条**：

    D7-09  基金线和期货线在销售人员管理上有什么共同要求
           → 选中 investor-classification-three-lines（0.541）
           → 该中 sales-staff-management-comparison（0.489）

    D7-12  期货和证券在违反适当性义务的行政处罚上有什么不同
           → 选中 securities-vs-futures-suitability（0.820）
           → 该中 penalty-comparison（0.658）

被选中的两条，标题都长这样：

    「证券、期货、基金三条业务线的投资者分类制度对照」
    「证券与期货适当性义务的法律规定对照」

**它们是一个"通用枢纽"**——凡是问证券/期货/基金 + 适当性的问题，
都跟它们的标题有很高的相似度。而它们真正有区分度的部分是触发词
（"销售人员管理" / "行政处罚"），但那部分在"标题 + 触发词拼成一句"
之后被稀释了。

这是关键词路那个坑的**语义版本**：上一次是"适当性义务"这 5 个字
赢了"行政处罚"这 4 个字；这一次是"证券与期货适当性义务"这串通用词
赢过了具体主题词。**同一个病，换了层皮。**

所以做这个对照：把"标题"和"触发词"**分开算相似度，取最高的那个**，
而不是拼成一句算平均。理由是——一条词条命中，靠的是**它有一个说法跟问题对上了**，
不是"它整段话平均起来像问题"。判据的形状要和打分方式一致。

用法（会调 embedding 接口，只调一次批量）：
    python 评测/_路由策略对照.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / '评测'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from sqlalchemy import select  # noqa: E402

from app.core.embeddings import embed_texts  # noqa: E402
from app.core.postgres import get_session_factory  # noqa: E402
from app.models.wiki_entry import WikiEntry  # noqa: E402
from app.rag.query_split import detect_lines  # noqa: E402
from app.rag.wiki_route import _bare, _cosine, is_comparison  # noqa: E402
from 评测集 import load_rows, pick  # noqa: E402

# 正确路由的人工标注。（同 工具/诊断词条路由.py，看清事实上的"该走哪条"。）
EXPECTED = {
    'D7-01': 'investor-classification-three-lines',
    'D7-03': 'securities-vs-futures-suitability',
    'D7-04': 'industry-association-ownership',
    'D7-05': 'risk-rating-comparison',
    'D7-06': 'investor-classification-three-lines',
    'D7-07': 'revisit-requirements-comparison',
    'D7-08': 'self-regulation-division',
    'D7-09': 'sales-staff-management-comparison',
    'D7-10': 'record-retention-comparison',
    'D7-11': 'matching-principles-comparison',
    'D7-12': 'penalty-comparison',
    'D7-13': 'industry-association-ownership',
    'D7-14': 'risk-catalogue-responsibility',
    'D7-15': 'lowest-risk-tolerance-investor',
}

# 关键词路的主题触发词最短长度（与 wiki_route 保持一致）。
MIN_TOPIC_TRIGGER = 3


def keyword_match(query: str, entries) -> tuple[str | None, float]:
    """复刻 wiki_route 里那条**关键词兜底**判据，用来做对照。

    复制一份而不是 import：兜底那条路现在只有在"拿不到向量"时才会走到，
    直接调用拿不到结果。但这恰恰是要测的东西——
    **它在有向量的场景下会不会比语义更准。**
    """

    best_slug, best_score = None, -1.0
    bare_query = _bare(query)
    asked_lines = set(detect_lines(query))
    for entry in entries:
        triggered = [
            str(trigger)
            for trigger in (entry.triggers or [])
            if len(str(trigger)) >= MIN_TOPIC_TRIGGER and _bare(str(trigger)) in bare_query
        ]
        if not triggered:
            continue
        overlap = len(set(entry.legal_lines or []) & asked_lines)
        score = overlap * 100 + float(sum(len(t) for t in triggered))
        if score > best_score:
            best_slug, best_score = entry.slug, score
    return best_slug, best_score


def best_match(query_vector, vectors_by_slug, strategy):
    """按某种策略给每条词条打分，返回 (slug, 分)。

    strategy:
      'joined'  —— 标题 + 触发词拼成一句（**现状**）
      'max'     —— 标题和每个触发词各算一次，取最高分
      'title'   —— 只看标题
      'triggers'—— 只看触发词（标题不参与）
    """

    best_slug, best_score = None, -1.0
    for slug, parts in vectors_by_slug.items():
        title_vector = parts.get('__title__')
        trigger_vectors = [v for k, v in parts.items() if k != '__title__']
        if strategy == 'joined':
            candidates = [parts['__joined__']]
        elif strategy == 'max':
            candidates = ([title_vector] if title_vector else []) + trigger_vectors
        elif strategy == 'title':
            candidates = [title_vector] if title_vector else []
        elif strategy == 'triggers':
            candidates = trigger_vectors
        else:
            raise ValueError(strategy)
        score = max((_cosine(query_vector, v) for v in candidates if v), default=-1.0)
        if score > best_score:
            best_slug, best_score = slug, score
    return best_slug, best_score


# 混合策略里"命中一个触发词"值多少分。
#
# 这个数**不是调出来的好数字，是量出来的**：语义分在 0.45~0.95 之间，
# 而"具体主题词逐字命中"这件事，本身就是一个很强的证据——
# 触发词是人工挑的、有区分度的词，不是碰巧共现的词。
# 0.05 × 词长：命中一个 6 字主题词值 0.30，足以翻转"通用枢纽词"带来的
# 0.1~0.2 的虚高，又不至于让一个碰巧命中的 3 字词把打分掀翻。
KEYWORD_BONUS_PER_CHAR = 0.05


def hybrid_match(query_vector, vectors_by_slug, entries, query, per_char=None):
    """语义分 + 触发词命中加成。

    **为什么走到"混合"这一步不是退回去。**

    关键词路单独用确实不行——这一点前面已经用三次失败证明过了。但那是
    **"关键词当唯一判据"**的失败，不是"关键词这件事没价值"的失败。
    这一轮的对照给出了明确的边界：

      · 语义**能**做：把问题大致归到哪个主题上（D7 里 11/14 直接对）；
      · 语义**不能**做：区分两条主题相邻的词条。看这两组：
            处罚对比   vs  适当性义务对照   （0.658 vs 0.820）
            销售人员管理 vs  投资者分类     （0.489 vs 0.541）
        它们被选中，不是因为更像，是因为对方标题里有一串**通用枢纽词**
        （"证券与期货""三条业务线"）。**主题相邻时，向量分不开。**

    而"销售人员管理""行政处罚"这种**具体主题词一旦逐字出现在问题里**，
    它就是一个比向量相似度更硬的证据。所以：**语义负责粗定位，
    关键词负责在相邻的词条之间裁决。** 两层各干各擅长的事。
    """

    per_char = KEYWORD_BONUS_PER_CHAR if per_char is None else per_char
    bare_query = _bare(query)
    best_slug, best_score = None, -1.0
    for entry in entries:
        parts = vectors_by_slug.get(entry.slug) or {}
        joined_vector = parts.get('__joined__')
        if not joined_vector:
            continue
        semantic = _cosine(query_vector, joined_vector)
        matched = [
            str(trigger)
            for trigger in (entry.triggers or [])
            if len(str(trigger)) >= MIN_TOPIC_TRIGGER
            and _bare(str(trigger)) in bare_query
        ]
        # 取**最长**那个，而不是求长度和。
        # 理由：求和会奖励"触发词多但是都很泛"的词条（自律管理 4 字 + 行业协会 4 字
        # 加起来 8 分，跟"交易所和行业协会"这个 8 字的**精确**命中打平）。
        # 最长命中衡量的是"这条词条有没有一个说法跟问题几乎逐字对上"，
        # 那才是我们要的证据。
        keyword = max((len(item) for item in matched), default=0)
        score = semantic + per_char * keyword
        if score > best_score:
            best_slug, best_score = entry.slug, score
    return best_slug, best_score


def main() -> int:
    session_factory = get_session_factory()
    with session_factory() as session:
        entries = list(
            session.execute(
                select(WikiEntry).where(WikiEntry.status == 'reviewed')
            ).scalars()
        )

    # 一次批量把所有要向量化的文本算完——embedding 是按条计费的，
    # 分开调用既慢又贵，而且没有意义。
    texts: list[str] = []
    layout: list[tuple[str, str]] = []  # (slug, key)
    for entry in entries:
        joined = f'{entry.title} {" ".join(entry.triggers or [])}'
        layout.append((entry.slug, '__joined__'))
        texts.append(joined)
        layout.append((entry.slug, '__title__'))
        texts.append(entry.title)
        for trigger in entry.triggers or []:
            layout.append((entry.slug, str(trigger)))
            texts.append(str(trigger))

    vectors = embed_texts(texts)
    vectors_by_slug: dict[str, dict] = {}
    for (slug, key), vector in zip(layout, vectors):
        vectors_by_slug.setdefault(slug, {})[key] = tuple(vector)

    rows = [row for row in load_rows() if pick(row, 'code') in EXPECTED]
    query_vectors = embed_texts([pick(row, 'question') for row in rows])

    strategies = ['joined', 'max', 'title', 'triggers', 'keyword', '混合']
    results = {name: [] for name in strategies}

    for row, query_vector in zip(rows, query_vectors):
        code = pick(row, 'code')
        expected = EXPECTED[code]
        wrong = []
        for name in strategies:
            if name == 'keyword':
                # 关键词路不需要向量，所以它**也是"能不能省掉一次 embedding"的候选**。
                slug, score = keyword_match(pick(row, 'question'), entries)
            elif name == '混合':
                slug, score = hybrid_match(
                    query_vector, vectors_by_slug, entries, pick(row, 'question')
                )
            else:
                slug, score = best_match(query_vector, vectors_by_slug, name)
            ok = slug == expected
            if not ok:
                wrong.append(f'{name}={slug}({score:.3f})')
            results[name].append(ok)
        mark = '✅' if not wrong else '❌'
        print(f'{mark} {code}  该中 {expected}')
        print(f'     {pick(row, "question")}')
        if wrong:
            for item in wrong:
                print(f'     ✗ {item}')

    print()
    print('=' * 78)
    print(f'{"策略":<12}{"选对 / 参加门":<16}准确率')
    for name in strategies:
        ok = sum(results[name])
        total = len(results[name])
        print(f'{name:<12}{f"{ok} / {total}":<16}{ok / total:.0%}')

    # ---- 敏感性检验：加成系数是不是"调出来的好数字" ----
    #
    # 这一步不能省。0.05 是我选的，如果只有 0.05 这一个值能到 14/14，
    # 那说明结果来自**对 14 道题过拟合**，换一批题就崩。
    # 反过来，如果一段区间都一样好，那说明结构本身在起作用。
    # （文档多样性那次就是这个道理：0.03 和 0.08 结果完全相同，
    #   说明那个 +2 题不是调参调出来的。）
    print()
    print('敏感性检验：每个触发词字符值多少分')
    print(f'{"系数":<8}{"选对":<8}准确率')
    for per_char in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.2, 0.3, 0.5, 1.0):
        ok = 0
        for row, query_vector in zip(rows, query_vectors):
            slug, _ = hybrid_match(
                query_vector, vectors_by_slug, entries, pick(row, 'question'), per_char
            )
            if slug == EXPECTED[pick(row, 'code')]:
                ok += 1
        print(f'{per_char:<8}{f"{ok} / {len(rows)}":<8}{ok / len(rows):.0%}')
    print()
    print('系数 0.0 = 纯语义（现状）。中间一段平坦 = 结构在起作用，不是调参。')
    print()
    print('注：这里只统计**过了门**的 14 条 D7（D7-02 没有对应词条，不参与）。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
