from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

"""拒答判定：**把"答不了"分成几种互不相同的情况，并且按确定性排序。**

## 为什么要把这一段从问答服务里抽出来

它原来是写在 `qa_service.ask()` 里的一段 if。抽出来有两个理由：

  1. 它是**唯一一个"明明检索到了，却决定不给答案"的地方**，
     而"为什么拒答"是产品里最需要能复盘的一件事；
  2. 它要被别的地方复用——「检索调试台」要显示"这个问题按当前配置会不会被拒"，
     标定工具要算"阈值取某个值时拒答比例是多少"。散在服务层就只能复制。

## 判据的顺序：确定性的先判，分数的最后判

这是这一层最重要的设计，也是这一轮改动的核心。

原来只有一条判据：**拿最高分跟阈值比**。它有三个问题：

**问题一：分数根本管不了"有没有"这类问题。**

    用户问："《证券期货投资者适当性管理办法》第八十条怎么规定的"

    这部办法一共只有 43 条。答案是**确定的不存在**——去库里按条号查一下，
    0 条命中，这件事没有任何不确定性。但分数阈值给出的答案是
    "最高分 0.31，低于阈值 0.5，所以相关度不足"——
    它把一个确定的事实说成了一个模糊的猜测。

    反过来也一样：问"证券法第八十八条"，条款直查已经把整条取出来了，
    分数却是 None（关系库取出来的东西没有相似度），
    于是**最硬的依据反而最容易被分数闸门拦掉**。这个坑踩过，见 `_best_score` 的老注释。

**问题二：分数阈值的量纲不稳定。**

    `hit['score']` 是个**混合字段**：重排生效时它是重排分（0~1），
    重排不可用时它是融合分（0.01 量级），只有一路召回时它是
    **BM25 分（10~50 量级）**。翻 2026-09-29 那天欠费期间的日志就能看到：
    同一个问题，向量路挂掉之后 `score` 记成了 47.87。
    阈值设成 0.5 的话，那天的所有查询都会"分数很高"、全部放行——
    **系统坏掉的那一天，拒答闸门恰好最松。**

**问题三：它把拒答原因混成了一句"相关度不足"。**

    用户看到"相关度不足"，该做什么？补资料？换问法？还是等系统恢复？
    这三种情况的行动完全不同，而分数阈值给不出区分。

所以判据按**确定性从高到低**排：

    1. 系统故障            → 拒，但明确说"是系统坏了，不是资料没有"（要重试）
    2. 条款直查：查不到     → 拒，确定性的"库里没有这一条"（要去补法规）
    3. 条款直查：查到了     → **答**，不经过阈值（这是最硬的依据）
    4. 词条命中            → **答**，不经过阈值（已编译的跨文档结论）
    5. 一条都没召回        → 拒，说明"库里可能没有这个主题"
    6. 混合检索：分数不足   → 拒，这才是唯一该由阈值管的情况

注意第 6 条的附加条件：**只有在重排分可用时才用阈值。**
阈值是在重排分（0~1）这个量纲上标定出来的，重排不可用时进入降级模式，
此时**不拿别的量纲去硬套**，而是照常作答、但在答案里标注"本次没有重排"。
理由：拿错量纲去拒答，等于用一个随机数决定要不要回答。
"""

# 拒答 / 放行的种类。存进日志，用来算"拒答正确率"和"过度拒答率"。
#
# 这两组指标必须成对看：只优化"该拒的拒了"，系统会退化成"什么都不答"，
# 那个版本的指标反而更好看。所以 kind 要分得足够细，否则
# "拒了多少"里混着"系统坏了"和"真的没有"，两个数字都失去意义。
KIND_OK = 'ok'
KIND_RETRIEVAL_ERROR = 'retrieval_error'      # 系统故障，不是知识缺口
KIND_CITATION_MISSING = 'citation_missing'    # 明确指出条号，库里没有
KIND_NO_HITS = 'no_hits'                      # 一条都没召回
KIND_LOW_SCORE = 'low_score'                  # 召回到了但重排分不足

# 依据的强度，也是给用户看的"这个答案有多硬"。
EVIDENCE_CITATION = 'citation_exact'   # 关系库按条号精确取出
EVIDENCE_WIKI = 'wiki'                 # 已编译的跨文档词条
EVIDENCE_RERANK = 'rerank'             # 重排分达标
EVIDENCE_DEGRADED = 'degraded'         # 重排不可用，按原顺序作答
EVIDENCE_NONE = 'none'


@dataclass
class RefusalDecision:
    """一次拒答判定的结果。"""

    refuse: bool
    kind: str
    evidence: str = EVIDENCE_NONE
    # 给用户看的、可以说出口的理由
    reason: str | None = None
    # 回答里要额外附带的提醒（不是拒答，但必须说清楚）
    note: str | None = None
    # 判定时实际用到的分数与阈值。**必须记下来**——
    # "同一批题目，两次跑结论不同"这类问题，只有靠它才能复盘。
    score: float | None = None
    score_kind: str | None = None
    threshold: float | None = None


def gate_score(hits: list[dict[str, Any]]) -> tuple[float | None, str]:
    """取出**用于阈值判定的那个分数**，并说明它是什么。

    只在命中里找 `rerank_score`，找不到就返回降级信号。

    为什么不在找不到的时候退回 `score`：`score` 是混合量纲（见模块开头）。
    退回它等于"用 BM25 的 47 分去比 0.5 的阈值"，
    结果是**所有查询全部放行**——一个静默的、只在故障期间出现的漏洞。
    """

    scores = [
        float(hit['rerank_score'])
        for hit in hits
        if isinstance(hit.get('rerank_score'), (int, float))
    ]
    if not scores:
        return None, 'unavailable'
    return max(scores), 'rerank'


def decide(
    *,
    query: str,
    hits: list[dict[str, Any]],
    wiki_entry: dict[str, Any] | None,
    citation: dict[str, Any] | None,
    error: str | None,
    threshold: float,
) -> RefusalDecision:
    """判定这次要不要拒答。参数全是检索结果的**事实**，不含判断。"""

    citation = citation or {}

    # ---- 1. 系统故障。它不是知识缺口，必须和"不知道"分开 ----
    if not hits and error:
        return RefusalDecision(
            refuse=True,
            kind=KIND_RETRIEVAL_ERROR,
            evidence=EVIDENCE_NONE,
            reason=f'检索服务暂时不可用（{error}）',
        )

    # ---- 2/3. 条款直查：**确定性判据**，不看分数 ----
    #
    # `usable` 的含义是"法规名和条号**都**解析出来了"。
    # 只解析出条号是不够的——"第二十九条"在库里命中 13 个不同文件，
    # 那时我们并不知道用户问的是哪一部，属于"没听懂"，不是"库里没有"。
    if citation.get('usable'):
        hit_count = int(citation.get('hit_count') or 0)
        title = citation.get('document_title') or '该法规'
        article = citation.get('article_number') or ''
        if hit_count > 0:
            return RefusalDecision(
                refuse=False,
                kind=KIND_OK,
                evidence=EVIDENCE_CITATION,
                note=f'本条依据《{title}》{article} 原文直接给出，未经过相似度排序。',
            )
        # 解析得清清楚楚，库里就是没有这一条 —— 这是**确定性结论**。
        return RefusalDecision(
            refuse=True,
            kind=KIND_CITATION_MISSING,
            evidence=EVIDENCE_NONE,
            reason=(
                f'已识别到《{title}》{article}，但知识库里没有收录这一条。'
                f'（法规名和条号都解析成功，这是确定结果，不是"没搜到"）'
            ),
        )

    # ---- 4. 词条：命中就是确定性的"有依据" ----
    if wiki_entry:
        return RefusalDecision(
            refuse=False,
            kind=KIND_OK,
            evidence=EVIDENCE_WIKI,
            note=(
                f'本条依据知识库词条「{wiki_entry.get("title")}」，'
                f'它是对多条法规的预先编译，未经过单片段相似度排序。'
            ),
        )

    # ---- 4b. 该有词条却没有：确定性地说明"缺什么" ----
    #
    # 这一条不拒答，只标注。理由要说清楚，否则下一个人一定会问"为什么不直接拒"：
    #
    #   · "没有词条"**不等于**"答不了"。有些对比题，两边条文都在库里，
    #     检索出来的原文拼起来也能回答，只是不如词条那种对照清楚；
    #   · 而"直接拒"是一个会**放大拒答率**的改动，它必须由过度拒答率的基线
    #     来决定，不能在这里顺手做掉。
    #
    # 所以现在做的是**把事实标出来**：这类问题在评测里只有 60% 能靠检索答对
    # （剩下一半是拼出来的对照），用户有权知道这次拿到的不是词条。
    note: str | None = None
    try:
        from app.rag.wiki_route import is_comparison

        if is_comparison(query):
            note = (
                '这个问题属于跨文档对照类，但知识库里还没有编译对应的词条；'
                '下面的回答只基于检索到的原文片段，可能不完整。'
            )
    except Exception:  # noqa: BLE001
        logger.exception('[REFUSAL] 对比类判定失败，跳过词条缺失提示')

    # ---- 5. 检索正常，但一条都没有 ----
    if not hits:
        return RefusalDecision(
            refuse=True,
            kind=KIND_NO_HITS,
            evidence=EVIDENCE_NONE,
            reason='检索正常执行，但没有召回任何片段',
        )

    # ---- 6. 阈值：**唯一由分数决定的情况**，而且必须是重排分 ----
    score, score_kind = gate_score(hits)
    if score is None:
        # 重排不可用（关掉了、或者调用失败退回了原顺序）。
        # 此时分数不是标定阈值时的那个量纲，**拿它比阈值等于掷骰子**。
        return RefusalDecision(
            refuse=False,
            kind=KIND_OK,
            evidence=EVIDENCE_DEGRADED,
            note=' '.join(
                filter(
                    None,
                    [
                        '本次未启用重排（或重排调用失败），结果按融合顺序给出，'
                        '相关性判断会比正常情况弱一些。',
                        note,
                    ],
                )
            ),
        )

    if score < threshold:
        return RefusalDecision(
            refuse=True,
            kind=KIND_LOW_SCORE,
            evidence=EVIDENCE_NONE,
            reason=(
                f'召回内容的相关度不足（重排分最高 {score:.4f}，阈值 {threshold:.4f}）'
            ),
            score=score,
            score_kind=score_kind,
            threshold=threshold,
        )

    return RefusalDecision(
        refuse=False,
        kind=KIND_OK,
        evidence=EVIDENCE_RERANK,
        note=note,
        score=score,
        score_kind=score_kind,
        threshold=threshold,
    )


__all__ = [
    'EVIDENCE_CITATION',
    'EVIDENCE_DEGRADED',
    'EVIDENCE_NONE',
    'EVIDENCE_RERANK',
    'EVIDENCE_WIKI',
    'KIND_CITATION_MISSING',
    'KIND_LOW_SCORE',
    'KIND_NO_HITS',
    'KIND_OK',
    'KIND_RETRIEVAL_ERROR',
    'RefusalDecision',
    'decide',
    'gate_score',
]
