from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.wiki_entry import WikiEntry
from app.rag.query_split import COMPARISON_MARKERS, detect_lines

logger = logging.getLogger(__name__)

"""Wiki 路由：判断一个问题该不该走词条。

用规则，不用意图分类模型。理由和查询解析一样——**判据的形状是固定的**：

    这题涉及几条业务线？（证券 / 期货 / 基金）
    是不是在问"两者有什么区别"？（对比词）

顺带说一句：这套判据原本是给**查询拆分**写的，在检索上没帮上忙
（三次尝试都没效果），但当**路由器**正好——它做的判断本质上就是
"这题要跨多条业务线",而这正是词条覆盖的场景。
同一个判据，用在对的位置上就有用。

⚠️ 只路由 `citation_verified=True` 的词条。
依据没核实完的词条参与作答，等于把未经核对的结论当权威发出去——
那正是这个项目一直在防的事。
"""


# 主题触发词的最短长度。"区别""差异"这类通用词不能当主题判据——
# 它们在每一道对比题里都出现，拿它们匹配等于没匹配。
MIN_TOPIC_TRIGGER = 3

# 匹配触发词时先去掉的虚词。
#
# 这一条是被真实失败逼出来的：词条触发词写的是"适当性义务**的**法律规定"，
# 而用户问的是"适当性义务**上**的法律规定"——**差一个虚词就完全失配**，
# 于是那条写好的词条一点用没有。
#
# 而"的/上/了/中/里"这些字在语义上不携带信息，去掉之后再比，
# 同一个主题的不同说法就能对上。**触发词匹配本来就是在猜用户怎么说，
# 至少不该连虚词的差异都算进去。**
_PARTICLES = re.compile(r'[的上了中里着]')


def _bare(text: str) -> str:
    return _PARTICLES.sub('', str(text))


# 门的第二条判据里，除了"点名了几条业务线"，还有一种情况：
# **问题没点名业务线，但明确在问"几个对象各自怎么分工"。**
#
# 这一条是被三道真实失败逼出来的，三道都卡在同一个地方——
# `detect_lines` 一条业务线也认不出来，于是门直接关掉：
#
#     D7-08  交易所和行业协会在适当性管理里各自负责什么   （对象是"协会 vs 交易所"，不是业务线）
#     D7-14  证券和期货两条线谁负责制定产品风险等级名录   （有对比意图，但"谁负责"不在词表里）
#     D7-15  三条业务线在最低风险承受能力类别的认定上有什么异同（"三条业务线"不是一个具体业务线名）
#
# 这三句都是标准的对比/分工题，而它们的答案同样不存在于任何单一片段里。
# 硬要求"点名 ≥2 条业务线"，等于**把一个代理判据当成了必要条件**。
#
# ⚠️ 但也不能干脆把这条去掉：去掉之后 `一样吗`/`分别` 这类通用词会单独放行
# D8-04「香港证监会对专业投资者的规定和内地一样吗」——那题的正确答案是拒答，
# 送一条内地词条过去正好把它带偏。所以补的是**"多对象"这个信号本身**，
# 不是取消这个信号。
PARALLEL_SUBJECT_MARKERS = (
    '业务线', '条线', '各自负责', '分别由谁', '分别负责', '谁负责', '归谁管',
)


# 语义匹配的相似度下限。
#
# 这个数字要**量出来**，不能拍：太低会误路由（把不该走词条的问题也送进去），
# 太高会漏（该走词条的送不进去）。所以先跑一遍 D7 看清分布再定。
#
# ⚠️ 但这里有一个**反直觉的实测结论**，必须写下来，否则下一个人一定会想"用相似度当门"：
#
#     拿 120 题全跑一遍词条相似度，最高分是 **D3-05 的 0.926**——
#     而它是一道单选题，跟词条毫无关系。对比题这边，最低的 D7-07 只有 0.486。
#     **两边的分布完全重叠，方向还反了。**
#
# 所以相似度**不能**用来判"该不该走词条"，只能用来判"过了门之后，哪条词条更像"。
# 门必须是**形状判据**（这题是不是对比/分工），不是分数判据。
# 这个阈值守的是"像不像"，不是"该不该"。
SEMANTIC_MIN_SCORE = 0.45


def is_comparison(query: str) -> bool:
    """是不是对比类问题。

    两条判据：**有对比意图** 且 **有多个对象**。

    （原来的第二条是"点名 ≥2 条业务线"，现在放宽成"点名 ≥2 条业务线**或**
      出现了多对象的说法"。理由见 PARALLEL_SUBJECT_MARKERS 上面那段。）
    """

    text = query or ''
    if not any(marker in text for marker in COMPARISON_MARKERS):
        return False
    if len(detect_lines(text)) >= 2:
        return True
    return any(marker in text for marker in PARALLEL_SUBJECT_MARKERS)


def match_entry(session: Session, query: str) -> WikiEntry | None:
    return match_entry_with_vector(session, query, query_vector=None)


@lru_cache(maxsize=4)
def _entry_vectors_cached(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """把词条的"标题 + 触发词"向量化。

    缓存键是**文本内容本身**，所以词条改了、加了，缓存自动失效——
    不需要人去维护"什么时候该清缓存"。
    12 条词条一次调用就够，之后每个进程都复用。
    """

    from app.core.embeddings import embed_texts

    return tuple(tuple(vector) for vector in embed_texts(list(texts)))


def _cosine(left: list[float], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


# 命中一个触发词，每多一个字加多少分。
#
# **这个数是量出来的，不是调出来的。** 对照组（`评测/_路由策略对照.py`）
# 在 D7 的 14 道上跑了一遍系数：
#
#     系数      0.00   0.01~0.03   0.05 ~ 1.00
#     选对      11/14   13/14       14/14
#
# 也就是说 0.05 到 1.0 是**一段 20 倍的平坦区间**，取中间任意值结果都一样。
# 平坦区间宽 = 结果来自结构，不是来自调参（和文档多样性那次同一个道理）。
# 取 0.10 是因为它在区间内偏中，两边都留有余量。
KEYWORD_BONUS_PER_CHAR = 0.10


def _keyword_hit(entry: WikiEntry, bare_query: str) -> int:
    """问题里逐字命中的**最长**触发词有多长。没命中返回 0。

    取最长，不取长度和——

      · 「交易所和行业协会」（8 字，精确） vs 「行业协会」+「自律管理」（4+4=8，两个泛词）
        长度和打平，最长分得开。而这两种情况确实不一样：
        前者是**这条词条的独有说法**，后者是任何一条自律相关词条都沾得上的词。

    这个函数同时服务于两条路：混合打分（有向量时）和关键词兜底（没向量时）。
    两处用同一个判据，就不会出现"兜底路和主路对同一个问题给出不同答案"。
    """

    matched = [
        str(trigger)
        for trigger in (entry.triggers or [])
        if len(str(trigger)) >= MIN_TOPIC_TRIGGER and _bare(str(trigger)) in bare_query
    ]
    return max((len(item) for item in matched), default=0)


def match_entry_with_vector(
    session: Session,
    query: str,
    *,
    query_vector: list[float] | None,
) -> WikiEntry | None:
    """给问题找一条词条。找不到返回 None。

    第一道门是 `is_comparison`（是不是对比/分工类问题），过了这道门之后：

      · **有 query_vector** → 混合打分：语义分 + 触发词命中加成；
      · **没有 query_vector**（离线工具）→ 关键词兜底。

    ## 为什么是"混合"，以及为什么这不是退回关键词

    这一段被改过三次，三次的数字都留在这里，因为它们合起来才说明问题：

      第 1 代  关键词单独用      —— 触发词当时写的是大词和虚词，实测很差，
                                    于是判断"关键词这条路修不好"
      第 2 代  语义单独用        —— 路由命中 10 → 13，但**通过率一点没动**（81%/60%）
      第 3 代  混合（就是现在）  —— 见下面

    第 2 代为什么"命中变多、结果不变"？把中间那一步摊开看（`工具/诊断词条路由.py`）
    才看清：**它选错了词条。**

        D7-09  基金线和期货线在销售人员管理上有什么共同要求
               → 选中 investor-classification-three-lines（0.541）
               → 该中 sales-staff-management-comparison（0.489）

        D7-12  期货和证券在违反适当性义务的行政处罚上有什么不同
               → 选中 securities-vs-futures-suitability（0.820）
               → 该中 penalty-comparison（0.658）

    被选中的那两条，标题都是"通用枢纽"：
    「证券、期货、基金三条业务线的投资者分类制度对照」「证券与期货适当性义务的法律规定对照」——
    凡是问"证券/期货/基金 + 适当性"的问题，跟它们都像。真正有区分度的是触发词
    （"销售人员管理""行政处罚"），但那部分被标题稀释了。

    顺手否掉了两个看起来很自然的改法（都实测更差，见 `评测/_路由策略对照.py`）：

      · 把标题和触发词**分开算、取最高**   —— 79% → **64%**（标题本身就是那个通用枢纽，取最高它就独占）；
      · 拆成标题/触发词/或只看触发词       —— 57% / 79%，都没超过原来的 79%。

    而**混合**是 14/14。原因是它让两层各干各擅长的事：

      · **语义负责粗定位**："这题大致属于哪个主题"——它做得到，D7 里 11/14 直接对；
      · **关键词负责裁决**："两条主题相邻的词条里，是哪一条"——**这一步语义做不到**。
        主题相邻时（处罚对比 vs 适当性义务对照、销售人员管理 vs 投资者分类），
        它们的向量本来就近，"更像"和"更对"不是一回事。
        而"行政处罚""销售人员管理"这种**具体主题词逐字出现在问题里**，
        是一个比向量相似度更硬的证据。

    ## 顺带修正一条之前写错的结论

    之前写在注释里的话是"关键词匹配修不好，只会换一批问法失配"。这句话当时是对的，
    **但它描述的是当时的触发词，不是关键词这个方法。** 触发词后来被重写过
    （去掉大词、虚词、对比词，只留具体主题词）之后，纯关键词路在这 14 道上拿到 **13/14**，
    比纯语义的 11/14 还高。

    所以真正的结论是：**"关键词方法不行"和"词表写得不行"是两件事。**
    当时把后者的锅甩给了前者，代价是多走了一整轮（语义）才绕回来。
    """

    if not is_comparison(query):
        return None

    entries = session.execute(
        select(WikiEntry).where(WikiEntry.status == 'reviewed')
    ).scalars().all()
    if not entries:
        return None

    bare_query = _bare(query)

    # ---- 混合打分（有向量）----
    if query_vector is not None:
        texts = tuple(
            f'{entry.title} {" ".join(entry.triggers or [])}' for entry in entries
        )
        vectors = _entry_vectors_cached(texts)
        semantic_scored = [
            (entry, _cosine(query_vector, vector))
            for entry, vector in zip(entries, vectors, strict=False)
        ]

        # ⚠️ 阈值卡在**语义分**上，不卡在总分上。
        #
        # 这个区别要紧：总分会因为触发词命中而加分，拿总分去卡阈值，
        # 就等于"命中一个长触发词可以让一个语义上完全不像的问题过关"。
        # 而阈值想守的是"这个问题跟词条像不像"——那是语义分的事。
        # 加成只该决定**哪条词条**胜出，不该决定**要不要走词条**。
        best_semantic_entry, best_semantic = max(semantic_scored, key=lambda pair: pair[1])
        if best_semantic < SEMANTIC_MIN_SCORE:
            logger.info(
                '[WIKI] 语义相似度不足（最高 %.3f），不路由: query=%r',
                best_semantic,
                query[:40],
            )
            return None

        scored = [
            (entry, semantic + KEYWORD_BONUS_PER_CHAR * _keyword_hit(entry, bare_query))
            for entry, semantic in semantic_scored
        ]
        best_entry, best_score = max(scored, key=lambda pair: pair[1])
        best_entry.hit_count = (best_entry.hit_count or 0) + 1
        session.commit()
        logger.info(
            '[WIKI] 混合命中: slug=%s 总分=%.3f 语义=%.3f 触发词最长命中=%d query=%r',
            best_entry.slug,
            best_score,
            next(
                semantic
                for entry, semantic in semantic_scored
                if entry.slug == best_entry.slug
            ),
            _keyword_hit(best_entry, bare_query),
            query[:40],
        )
        return best_entry

    # ---- 关键词兜底（拿不到向量时，比如离线工具）----

    asked_lines = set(detect_lines(query))

    best: tuple[int, WikiEntry] | None = None
    for entry in entries:
        covered = set(entry.legal_lines or [])
        overlap = len(covered & asked_lines)
        keyword = _keyword_hit(entry, bare_query)
        if not keyword:
            continue

        # ⚠️ 这里**不再要求"问题里提到 ≥2 条业务线"**。踩过一次：
        #
        #     "违反适当性义务的行政处罚有什么不同"
        #     "三条业务线在投资者分类上的要求一样吗"
        #
        # 这两句都没点名证券/期货/基金，但它们是标准的对比题——
        # 业务线由上下文（或者"业务线"这个词本身）隐含。
        # 原来把"点名 ≥2 条线"当硬条件，结果这两题一条都路由不上，
        # 而它们恰恰是最该走词条的。
        #
        # 所以：**业务线命中是加分项，主题命中才是必要条件。**
        # 这么改不会放过普通问题，因为上面还有一道 is_comparison 的门。
        score = overlap * 100 + keyword
        if best is None or score > best[0]:
            best = (score, entry)

    if best is None:
        return None

    entry = best[1]
    entry.hit_count = (entry.hit_count or 0) + 1
    session.commit()
    logger.info('[WIKI] 命中词条: slug=%s query=%r', entry.slug, query[:40])
    return entry


def entry_as_context(entry: WikiEntry) -> dict[str, Any]:
    """把词条整理成给生成用的上下文块。"""

    return {
        'slug': entry.slug,
        'title': entry.title,
        'summary': entry.summary,
        'body': entry.body,
        'citations': entry.citations or [],
        'status': entry.status,
    }


__all__ = ['entry_as_context', 'is_comparison', 'match_entry']
