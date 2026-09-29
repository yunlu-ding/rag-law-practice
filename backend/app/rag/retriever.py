from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.embeddings import embed_query
from app.core.vector_store import get_vector_store
from app.rag.bm25_index import get_bm25_index
from app.rag.reranker import rerank
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

"""混合检索。

流程：

    问题 ─┬─ 向量路：embedding → 余弦相似 → Top-N
          └─ 关键词路：BM25 → Top-N
                   ↓
              RRF 融合（按名次融合）
                   ↓
              取前 candidate_k 条
                   ↓
              重排（可开关，失败退回原顺序）
                   ↓
              Top-K

**为什么用 RRF（倒数排名融合），而不是把两路分数加权相加：**
向量相似度和 BM25 分数的量纲完全不同，加权就得先归一化，
而归一化方式本身又是一个需要调参的主观选择。
RRF 只用名次：1/(k + rank)。它天然免疫量纲问题，也没有需要标定的权重。
代价是丢掉了分数的绝对信息——在这个阶段，稳健比精细重要。

**每一步的耗时都单独记录**，因为"慢"和"不准"是两个问题，
必须能分开看。检索调试台会把它们显示出来。
"""

# RRF 里的平滑常数。取 60 是文献里的常用值：
# 它让"第 1 名和第 2 名"的差距不至于过大，避免单路结果压倒性主导。
RRF_K = 60


@dataclass
class RetrievalOutcome:
    """一次检索的完整结果，含过程数据。

    过程数据和结果一样重要：出问题时你要能回答
    "是向量路没召回到，还是关键词路没召回到，还是重排把它压下去了"。
    只返回最终结果的话，这三个问题的答案就都没有了。
    """

    query: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    top_k: int = 5
    candidate_k: int = 0
    rerank_enabled: bool = False
    vector_hit_count: int = 0
    bm25_hit_count: int = 0
    fused_count: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    log_id: str | None = None
    citation: dict[str, Any] | None = None
    exact_hit_count: int = 0
    sub_queries: list[str] = field(default_factory=list)
    covered_lines: list[str] = field(default_factory=list)
    wiki_entry: dict[str, Any] | None = None
    # 词条**原始依据**的那些条款，从关系库取出来的真实切片。
    #
    # ⚠️ 它**故意不进 `hits`**，因为 `hits` 是检索层的成绩单
    # （评测判"锚点有没有被召回"看的就是它）。词条依据是预编译层的产物，
    # 把它塞进 hits 会让检索层的分数虚高——那是把两层的东西混在一起算。
    #
    # 它的用途只有一个：**让生成层有真实片段可引用**。
    # 起因是一个真实失败：模型根据词条答了"证券法第一百九十八条、期货法第一百三十五条"
    # 的罚款幅度，**却引用为空**——因为词条是作为另一个块给它的，没有编号可填。
    # 于是"给结论、没引用"就这么混过去了，用户没法核对。
    wiki_evidence: list[dict[str, Any]] = field(default_factory=list)
    # 两条路都空的时候，向量库里到底有多少条。只用于把
    # "库里真的没有" 和 "检索链路坏了" 分开（见 _flag_silent_empty）。
    vector_total: int | None = None


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    db: Session | None = None,
) -> RetrievalOutcome:
    """执行一次混合检索。

    `db` 是可选的：不传就只跑"向量 + 关键词"这条基本链路（纯检索，不碰关系库）。
    传了它，才会启用**条款级精确直查**——因为那一步要按条号去关系库里精确取数。
    """

    settings = get_settings()
    cleaned = (query or '').strip()
    outcome = RetrievalOutcome(
        query=cleaned,
        top_k=top_k,
        rerank_enabled=bool(settings.rerank_enabled),
    )
    if not cleaned:
        return outcome

    # 开了重排就粗筛一个更大的池子，交给重排去精选；
    # 没开重排就按 top_k 的几倍召回即可，池子开大没有意义。
    candidate_k = (
        max(settings.rerank_candidate_k, top_k)
        if outcome.rerank_enabled
        else max(top_k * 4, 10)
    )
    outcome.candidate_k = candidate_k

    # ---- 向量路 ----
    started = time.perf_counter()
    query_vector: list[float] | None = None
    try:
        vector = embed_query(cleaned)
        # 留在手里：Wiki 路由要用它做语义匹配。
        # **复用同一个向量，不额外调一次 embedding**——
        # 这是"语义匹配不增加成本"能成立的原因。
        query_vector = vector
        vector_hits = get_vector_store().search(vector=vector, top_k=candidate_k)
    except Exception as exc:  # noqa: BLE001
        # 单路失败不中断整条检索：另一路可能仍然能给出可用结果。
        # 但要把原因记下来——静默降级是上一个项目里最难查的一类问题。
        logger.exception('[RETRIEVE] 向量路失败: query=%r', cleaned)
        vector_hits = []
        outcome.error = f'向量路失败：{type(exc).__name__}: {exc}'
    outcome.timings_ms['vector'] = int((time.perf_counter() - started) * 1000)
    outcome.vector_hit_count = len(vector_hits)

    # ---- 关键词路 ----
    started = time.perf_counter()
    try:
        bm25_hits = get_bm25_index().search(cleaned, top_k=candidate_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception('[RETRIEVE] 关键词路失败: query=%r', cleaned)
        bm25_hits = []
        outcome.error = (outcome.error or '') + f' 关键词路失败：{type(exc).__name__}: {exc}'
    outcome.timings_ms['bm25'] = int((time.perf_counter() - started) * 1000)
    outcome.bm25_hit_count = len(bm25_hits)

    # ---- 两条路都是 0：必须分辨"真的没有"和"链路有问题" ----
    #
    # 这一段是被一次真实故障逼出来的，而且它暴露的是**一类**问题。
    #
    # 现象：用户问"证券期货投资者适当性管理办法第二十九条怎么规定的"，
    # 答案是无法判断。日志里那一次的记录是——
    #
    #     向量=0 关键词=0 融合=0 返回=0    error=None
    #
    # 三个 0 加上"没有报错"，看起来就是"知识库里没有这个主题"。
    # 但那个问题不但库里有，而且是**条款直查**能精确命中的一类。
    # 真正发生的是：那一刻两条检索路都返回了空，**而它们都没报错**。
    #
    # 这两件事对用户的意义完全不同：
    #   · "库里没有"   → 他去补资料（可能补的是一份**已经有了**的文件）；
    #   · "链路坏了"   → 他重试或去看向量库状态。
    #
    # 所以这里要判一次。判据不是"猜"，是用一个**不变量**：
    # 向量检索那边没有分数下限，**集合非空时余弦检索必然返回 top_k 条**
    # （唯一的空返回路径是"集合不存在"）。所以
    # "集合里有 N 条却一条都没召回"在正常情况下不可能发生。
    if not vector_hits and not bm25_hits:
        _flag_silent_empty(outcome, db)

    # ---- 查询拆分：只用来**扩大候选池** ----
    #
    # 位置很关键：它加的是候选，不改排序。重排仍然用**原问题**，
    # 因为"这两条业务线有什么区别"这个意图是原问题才有的，
    # 子查询（"证券 证券和期货…"）反而说不太清。
    #
    # 代价也小：只多几次向量化和关键词检索（都是便宜的），
    # 重排的调用次数**不增加**——候选池大小由 candidate_k 封顶，与原来一致。
    extra_sources: list[tuple[str, list[dict[str, Any]]]] = []
    sub_queries: list[str] = []
    line_pool: dict[str, list[dict[str, Any]]] = {}
    if settings.query_split_enabled:
        try:
            from app.rag.query_split import split_query

            sub_queries = split_query(cleaned)
        except Exception:  # noqa: BLE001
            logger.exception('[RETRIEVE] 查询拆分失败，按原问题检索: query=%r', cleaned)
            sub_queries = []

        for sub_query in sub_queries:
            label = sub_query.split(' ', 1)[0]
            started = time.perf_counter()
            try:
                sub_vector = embed_query(sub_query)
                line_vector = get_vector_store().search(vector=sub_vector, top_k=candidate_k)
                line_bm25 = get_bm25_index().search(sub_query, top_k=candidate_k)
                extra_sources.append((f'vector·{label}', line_vector))
                extra_sources.append((f'bm25·{label}', line_bm25))
                # 单独留一份"这条业务线自己召回了什么"，
                # 后面要在这份池子里挑一条保底，而不是从融合结果里挑。
                line_pool[label] = list(line_vector) + list(line_bm25)
            except Exception:  # noqa: BLE001
                logger.exception('[RETRIEVE] 子查询检索失败: sub=%r', sub_query)
            outcome.timings_ms['split'] = outcome.timings_ms.get('split', 0) + int(
                (time.perf_counter() - started) * 1000
            )

    outcome.sub_queries = sub_queries

    # ---- Wiki 词条通道 ----
    #
    # 走的是"路由"，不是"检索"。命中词条时它**不混进 hits**，
    # 而是单独挂在 outcome 上——原因是词条不是切片：
    # 它没有 chunk_id，混进 hits 会让引用还原指向一个不存在的片段。
    if settings.wiki_enabled and db is not None:
        try:
            from app.rag.wiki_route import entry_as_context, match_entry_with_vector

            entry = match_entry_with_vector(db, cleaned, query_vector=query_vector)
            if entry is not None:
                outcome.wiki_entry = entry_as_context(entry)
                outcome.wiki_evidence = _wiki_evidence(db, outcome.wiki_entry)
        except Exception:  # noqa: BLE001
            logger.exception('[RETRIEVE] Wiki 路由失败，按纯 RAG 继续: query=%r', cleaned)

    # ---- 融合 ----
    fused = reciprocal_rank_fusion(
        vector_hits, bm25_hits, limit=candidate_k, extra=extra_sources
    )
    outcome.fused_count = len(fused)

    # ---- 条款级精确直查 ----
    #
    # 放在融合之后、重排之前：它不参与排序，只负责"把精确命中的那几条
    # 捞进候选池，并记下它们是精确命中"。真正置顶发生在重排之后。
    exact_hits: list[dict[str, Any]] = []
    if db is not None:
        exact_hits = _try_citation_lookup(db, cleaned, outcome)

    # ---- 重排 ----
    #
    # ⚠️ 这里要**多要一些**，不能直接只要 top_k 条。
    #
    # 这是踩过的坑：多样性处理写在了重排之后，但重排只返回 top_k 条，
    # 于是多样性只能在 5 条里挑 —— 而"被同一份文件挤掉的正确答案"
    # 往往排在重排的第 6~20 名，**我的代码根本看不到它**。
    # 实测表现是改前改后数字一模一样（75% vs 75%），
    # 看起来像"这个改动没用"，实际是"这个改动没生效"。
    #
    # 代价几乎为零：重排接口本来就是拿全部候选去打分、再取前 N，
    # 多返回几条不增加调用量。
    dense_k = top_k
    if float(settings.rerank_diversity_penalty or 0.0) > 0:
        dense_k = max(top_k * 4, top_k + 10)

    started = time.perf_counter()
    final_hits = rerank(cleaned, fused, top_k=dense_k)
    outcome.timings_ms['rerank'] = int((time.perf_counter() - started) * 1000)

    if exact_hits:
        final_hits = _merge_exact_hits(exact_hits, final_hits, top_k=top_k)

    # ---- 每条业务线保底一个位置 ----
    #
    # 这是诊断之后改的方向。原来以为"对比类问题的第二个业务线不在候选池里"，
    # 实测发现候选池里有，**问题在重排**：它给《问答》里泛泛而谈的段落
    # 0.37 分，给《证券法》第八十八条、《期货和衍生品法》第五十条只有 0.15。
    #
    # 这不完全是重排的错——**没有哪一段单独回答了"两者有什么区别"**，
    # 交叉编码器给"部分相关"打低分是合理的。真正的问题是我们在用
    # "选最像的一段"的办法，去回答一个"需要两边都在"的问题。
    #
    # 所以：让每条业务线各自在自己的候选里挑一条最好的，**保底进结果**。
    # 重排仍然决定每条线内部谁是第一，只是不再让某一条线整条消失。
    if line_pool:
        forced = _line_coverage(
            sub_queries,
            line_pool,
            min_score=float(settings.query_split_min_score or 0.0),
        )
        if forced:
            forced_ids = {str(hit.get('chunk_id')) for hit in forced}
            final_hits = forced + [
                hit for hit in final_hits if str(hit.get('chunk_id')) not in forced_ids
            ]
            outcome.covered_lines = [hit.get('article_number') for hit in forced]

    final_hits = _diversify(
        final_hits,
        top_k=top_k,
        penalty=float(settings.rerank_diversity_penalty or 0.0),
        protect=int(settings.rerank_diversity_protect or 0),
    )

    document_meta = _load_document_meta(db, final_hits)
    outcome.hits = [_with_effect_labels(hit, document_meta) for hit in final_hits]
    outcome.exact_hit_count = len(exact_hits)
    logger.info(
        '[RETRIEVE] 完成: query=%r 向量=%s 关键词=%s 融合=%s 返回=%s 耗时=%s',
        cleaned,
        outcome.vector_hit_count,
        outcome.bm25_hit_count,
        outcome.fused_count,
        len(final_hits),
        outcome.timings_ms,
    )
    return outcome


def _flag_silent_empty(outcome: RetrievalOutcome, db: Session | None) -> None:
    """两条路都空的时候，判一次"是真没有，还是链路坏了"。

    为什么值得多花一次 RPC：**这两句话给用户的行动指引是相反的。**
    说"库里没有"，他会去补一份其实已经有的文件；说"链路坏了"，他会重试。
    而错误的那个说法（"库里可能没有"）恰恰是听起来最合理的那个。

    判据用的是不变量，不是猜测：

      · 向量集合**非空**时，余弦检索必然返回 top_k 条（那边没有分数下限，
        唯一的空返回路径是"集合不存在"）。所以"有 N 条却召回 0 条"
        = 链路异常，不可能是"库里没有"。
      · 向量集合**为空**、而关系库里还有切片，那是**两个库不同步**——
        同样是系统问题，不是知识缺口，而且它只能靠重建向量库来修。

    这个探测只在"两条路都空"时跑，正常查询一次都不会触发。
    """

    try:
        from app.core.vector_store import get_vector_store

        total = get_vector_store().count()
    except Exception as exc:  # noqa: BLE001
        # 连探测都失败了 —— 那更说明是链路问题，而不是知识库里没有。
        logger.exception('[RETRIEVE] 两条路都空，且向量库探测失败')
        outcome.error = (
            f'向量库探测失败（{type(exc).__name__}: {exc}）——'
            f'本次没有召回任何内容，是检索链路的问题，不代表知识库里没有'
        )
        return

    outcome.vector_total = total

    if total > 0:
        outcome.error = (
            f'向量库里有 {total} 条向量，但本次一条都没召回 —— '
            f'这是检索链路异常，不是知识库里没有'
        )
        logger.error(
            '[RETRIEVE] 静默空召回: 向量库 %s 条，两条路却都是 0。query=%r',
            total,
            outcome.query,
        )
        return

    # 向量库是空的。再看关系库有没有切片——有就是"两个库不同步"。
    if db is None:
        return
    try:
        from sqlalchemy import func, select

        from app.models.chunk import Chunk

        chunks = int(db.execute(select(func.count()).select_from(Chunk)).scalar() or 0)
    except Exception:  # noqa: BLE001
        logger.exception('[RETRIEVE] 统计切片数失败')
        return

    if chunks > 0:
        outcome.error = (
            f'关系库里有 {chunks} 个切片，但向量库里一条都没有 —— '
            f'两个库不同步，需要重建向量库（工具/重建向量库.py）'
        )
        logger.error('[RETRIEVE] 向量库为空但关系库有 %s 个切片', chunks)


def _try_citation_lookup(
    db: Session,
    query: str,
    outcome: RetrievalOutcome,
) -> list[dict[str, Any]]:
    """解析查询里的(法规名, 条号)，命中就去关系库精确取出该条的全部切片。

    这一步**只负责取数，不负责排序**。它先于重排发生，
    是因为需要它记下"哪几条是精确命中"；真正把它们放到最前面是在重排之后
    （见 _merge_exact_hits 的说明）。

    任何一步失败都不影响主链路——解析不出来就是"这次没有额外命中"，
    静默退回普通检索。**这是个增益，不是依赖。**
    """

    from app.rag.citation import load_document_refs, lookup_article, parse_citation

    try:
        citation = parse_citation(query, load_document_refs(db))
    except Exception:  # noqa: BLE001
        logger.exception('[RETRIEVE] 查询解析失败，跳过条款直查: query=%r', query)
        return []

    outcome.citation = {
        'article_number': citation.article_number,
        'article_int': citation.article_int,
        'document_id': citation.document_id,
        'document_title': citation.document_title,
        'matched_chars': citation.document_matched_chars,
        'candidates': citation.article_candidates,
        'usable': citation.usable,
        'reason': citation.reason,
    }

    if not citation.usable:
        logger.info('[RETRIEVE] 未做条款直查: %s', citation.reason)
        return []

    try:
        hits = lookup_article(
            db,
            document_id=citation.document_id,
            article_int=citation.article_int,
        )
    except Exception:  # noqa: BLE001
        logger.exception('[RETRIEVE] 条款直查失败: query=%r', query)
        return []

    outcome.citation['hit_count'] = len(hits)
    return hits


def _line_coverage(
    sub_queries: list[str],
    line_pool: dict[str, list[dict[str, Any]]],
    *,
    min_score: float,
) -> list[dict[str, Any]]:
    """每条业务线各自挑一条最好的，用来保底进入结果。

    关键在于**用子查询去打分，而不是用原问题**：
    原问题问的是"两者有什么区别"，它对任何一段单独的法条都只能给"部分相关"；
    而"期货 原问题"这句子查询问的是"期货这一边怎么规定的"，
    它对期货那一侧的法条能给出正常的判断。

    所以每一条业务线内部的排序仍然交给重排（它擅长这个），
    只是不再让"跨线比较"这个整体判断去压扁每一条线自己的代表。
    """

    forced: list[dict[str, Any]] = []
    for sub_query in sub_queries:
        label = sub_query.split(' ', 1)[0]
        pool = line_pool.get(label) or []
        if not pool:
            continue
        picked = rerank(sub_query, pool, top_k=1)
        if not picked:
            continue
        if float(picked[0].get('score') or 0.0) < min_score:
            # 这条线里也没有像样的东西。硬塞一个不相干的切片，
            # 只会把它当成"依据"喂给模型——宁可空着。
            logger.info('[RETRIEVE] %s 线没有达到下限的结果，不保底', label)
            continue
        hit = dict(picked[0])
        hit['retrieval_sources'] = list(hit.get('retrieval_sources') or []) + [
            f'coverage·{label}'
        ]
        forced.append(hit)
    return forced


def _merge_exact_hits(
    exact_hits: list[dict[str, Any]],
    ranked_hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """把精确命中的条款放在最前，其余按重排结果补齐。

    为什么是"置顶"而不是"加权"：

    条号是**标识符**，不是相似度。"这一段是不是第二十九条"只有是与否，
    没有"有 0.8 像"。把它做成一个加权项，就等于让它继续去和语义分数争——
    而实测里它争不过：重排给正确答案 0.4225，给一个无关规章里
    "本办法自2007年8月1日起施行"的第二十九条 0.4382。

    所以这里是一条硬规则：**精确命中的，不许被挤出去。**

    另外，精确命中可能有**多片**——一条法规常被分页切成两片（后半段没有条号，
    靠 article_number 继承父条号）。这些片全部保留，不做截断：
    用户问"第X条怎么规定的"，要的是完整那一条，不是最像的一片。
    """

    seen = {str(hit.get('chunk_id')) for hit in exact_hits}
    remainder = [
        hit for hit in ranked_hits if str(hit.get('chunk_id')) not in seen
    ]
    # 精确命中自己就超过 top_k 时，宁可多给几条也不截断那一条法规。
    keep = max(0, top_k - len(exact_hits))
    return exact_hits + remainder[:keep]


def _wiki_evidence(
    db: Session,
    wiki_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """把词条「依据」里那些条款，从关系库取出真实切片。

    起因是一个真实失败：模型依据词条答出了"《证券法》第一百九十八条、
    《期货和衍生品法》第一百三十五条"的罚款幅度，**却一条引用都没有**。

    根因不是它偷懒，是**结构上没有编号可填**——词条是作为单独一个块
    【知识库词条】交给它的，不在【检索片段】的编号体系里；
    而提示词又要求「依据片段」只能填检索片段的编号。于是它只能空着。

    所以修法不是逼它编一个引用，而是**把词条的原始依据变成真实片段**：
    词条的 citations 里写着（文档名 + 条款号），而这些条款本来就在库里。

    两个刻意的边界：

      · **不进 `hits`**（见 `wiki_evidence` 字段的说明）——检索层的成绩单
        不能因为预编译层的产物而虚高；
      · 取不到就跳过，**不报错、不阻断**。词条的依据清单是人工核对过的，
        但语料版本变了、文档标题对不上，都有可能取不到——
        那种情况下最差只是回到"这次没有可引用的原始条款"，
        不能让整个问答挂掉。
    """

    from app.models.document import Document
    from app.rag.citation import lookup_article
    from app.rag.splitters.legal import chinese_number_to_int

    evidence: list[dict[str, Any]] = []
    for citation in wiki_entry.get('citations') or []:
        title = str(citation.get('文档') or '').strip()
        article = str(citation.get('条款') or '').strip()
        number = chinese_number_to_int(article)
        if not title or number is None:
            continue
        try:
            # 词条的 citations 存的是**文档标题**（不是文件名），所以两个都试。
            document = db.execute(
                select(Document)
                .where((Document.title == title) | (Document.filename == title))
                .limit(1)
            ).scalar_one_or_none()
            if document is None:
                logger.info('[RETRIEVE] 词条依据对不上文档: %s', title)
                continue
            chunks = lookup_article(db, document_id=document.id, article_int=number)
        except Exception:  # noqa: BLE001
            logger.exception('[RETRIEVE] 取词条依据失败: %s %s', title, article)
            continue
        for chunk in chunks:
            chunk = dict(chunk)
            chunk['retrieval_source'] = 'wiki_evidence'
            chunk['retrieval_sources'] = ['wiki_evidence']
            # 词条依据同样是"某份法规的某一条"，所以该带的元数据要带全——
            # 少了效力层级，提示词里"必须说明依据出自哪一层"就落不了地。
            chunk = _with_effect_labels(
                chunk,
                {
                    str(document.id): {
                        'doc_number': document.doc_number,
                        'issued_date_label': _cn_date(document.issued_date),
                        'effective_date_label': _cn_date(document.effective_date),
                    }
                },
            )
            evidence.append(chunk)

    if evidence:
        logger.info(
            '[RETRIEVE] 词条依据已转成可引用片段: %s 片（来自 %s 条依据）',
            len(evidence),
            len(wiki_entry.get('citations') or []),
        )
    return evidence


def _cn_date(value: Any) -> str | None:
    """把日期渲染成"2017年7月1日"。

    为什么要转成人话：模型看到的应该是人读得懂的东西。
    `date(2017, 7, 1)` 这种写法它当然也能理解，但**用户最后看到的引用里
    会带上这一串**，而"2017-07-01"在一份中文合规材料里是突兀的。
    在检索层一次转好，下游（提示词、引用展示）都不用再各转一遍。
    """

    if value is None:
        return None
    try:
        return f'{value.year}年{value.month}月{value.day}日'
    except AttributeError:
        return str(value) or None


def _load_document_meta(
    db: Session | None,
    hits: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """按 document_id 取文档级元数据（公布日期 / 施行日期 / 文号）。

    为什么在这里补，而不是**预先塞进向量库**：

    这些字段本来就在关系库里，而且是**人工核对过**的（`语料元数据核对.json`）。
    塞进向量库意味着以后每改一次元数据，都要重建索引、再花一次 embedding 的钱；
    而挂在检索结果上，一次查询只多一条 `WHERE id IN (...)`，成本可以忽略。

    **这条是被一次真实的失败逼出来的**：用户问"《证券期货投资者适当性管理办法》
    什么时候开始施行的"，系统回答"未在检索片段中直接写出施行日期"——
    它没有说错，**它说的是实话**：施行日期早就抽出来了、就存在文档元数据里，
    只是没有跟着检索结果一起交给它。于是三道同类题全部答不出来。
    """

    if db is None or not hits:
        return {}

    ids = {str(hit.get('document_id')) for hit in hits if hit.get('document_id')}
    if not ids:
        return {}

    try:
        from app.models.document import Document

        rows = db.execute(
            select(
                Document.id,
                Document.doc_number,
                Document.issued_date,
                Document.effective_date,
            ).where(Document.id.in_(ids))
        ).all()
    except Exception:  # noqa: BLE001
        # 补元数据失败不该影响检索本身——最多是回答里少了日期，
        # 而不是这次检索作废。
        logger.exception('[RETRIEVE] 读取文档元数据失败')
        return {}

    return {
        str(row[0]): {
            'doc_number': row[1],
            'issued_date_label': _cn_date(row[2]),
            'effective_date_label': _cn_date(row[3]),
        }
        for row in rows
    }


def _with_effect_labels(
    hit: dict[str, Any],
    document_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """给检索结果补上"效力层级 / 效力状态"的中文标签，以及**分数的来源**。

    为什么要在这里补，而不是让提示词模块自己翻译：

    提示词里有条硬规则——回答"是否违规"时**必须说明依据出自哪一层效力**。
    而模型拿到的只有 `legal_level="department_rule"` 这样的机器值。
    如果让它自己去理解这个值，它只能猜；不给它这个值，它就只能编。

    所以：**在把片段交给模型之前，把机器值翻成人话。**
    这是"数据在哪一层就处理好哪一层"的做法——检索层知道层级，
    就别让提示词层去反推。

    顺带补一个 `score_kind`，它解决的是另一件容易出事的事：

        `score` 是一个**混合量纲**的展示字段。重排生效时它是重排分（0~1）；
        重排不可用时它是融合分（0.01 量级）；只有一路召回时它是
        BM25 分（10~50 量级）。翻 2026-09-29 欠费期间的日志能直接看到：
        同一个问题，向量路挂掉之后 `score` 变成了 47.87。

    后果不是显示难看，而是**任何拿 `score` 去跟阈值比的地方都会在故障期间失效**。
    所以这里把来源显式标出来，判定的地方只认 `rerank`。
    保留 `score` 不改语义，是因为「检索调试台」在展示它，而展示混合值是合理的——
    人看到 47.87 会去问"为什么这么高"，代码不会问。
    """

    from app.rag.metadata import VALIDITY_LABELS, level_label

    enriched = dict(hit)
    enriched['legal_level_label'] = level_label(hit.get('legal_level'))
    enriched['validity_label'] = VALIDITY_LABELS.get(
        str(hit.get('validity') or 'effective'), '效力未知'
    )

    # 文档级的日期与文号。它们回答的是"这份法规什么时候公布、什么时候施行"——
    # 而这类问题**在正文里往往找不到**（施行日期写在公告落款，不在条文里），
    # 所以它不是可有可无的装饰，是唯一能回答那一类问题的东西。
    meta = (document_meta or {}).get(str(hit.get('document_id'))) or {}
    for field in ('doc_number', 'issued_date_label', 'effective_date_label'):
        value = meta.get(field) or hit.get(field)
        if value:
            enriched[field] = value

    sources = hit.get('retrieval_sources') or []
    if sources == ['exact']:
        enriched['score_kind'] = 'exact'
    elif hit.get('rerank_score') is not None:
        enriched['score_kind'] = 'rerank'
    else:
        # 重排没生效（关掉了、或调用失败退回了原顺序）。
        # 这时 `score` 是融合分或单路分数，量纲不是阈值标定时用的那个。
        enriched['score_kind'] = 'fused'
    return enriched


def _diversify(
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    penalty: float,
    protect: int,
) -> list[dict[str, Any]]:
    """让结果里多出现几份不同的文件，而不是被一份文件占满。

    机制：同一份文件已经选过的，再选时**从分数里扣掉一个惩罚量**。
    惩罚是"每重复一次扣一点"，所以第 2 个同文件切片只扣一点、
    第 3 个扣两点——**不是一刀切禁止同文件**，而是让"另一份文件里稍微差一点"
    有机会挤进来。

    为什么前 `protect` 名不动：
    它们是最强证据，其中往往就有答案所在的那一条。为了多样性去动前两名，
    等于拿"必然正确的依据"换"可能相关的别的文件"，方向反了。
    重复带来的问题主要在**尾部**：3~5 名如果全是同一份文件的邻居，
    边际信息量很低，不如换成另一份相关的文件。

    这条改动同时针对基线里的两个现象：

      · 排头的是《适当性管理办法》问答，把真正的法规条文挤出去（占 14/28）；
      · "对比两条业务线"的问题，两份需要的文件只来了一份（D7 只有 20%）。
    """

    if penalty <= 0 or len(hits) <= 1:
        return hits

    protect = max(0, min(protect, len(hits)))
    chosen = list(hits[:protect])
    remaining = list(hits[protect:])

    def key_of(hit: dict[str, Any]) -> str:
        return str(hit.get('document_id') or hit.get('filename') or '')

    used: dict[str, int] = {}
    for hit in chosen:
        used[key_of(hit)] = used.get(key_of(hit), 0) + 1

    while remaining and len(chosen) < top_k:
        best_index = 0
        best_value: float | None = None
        for index, hit in enumerate(remaining):
            repeated = used.get(key_of(hit), 0)
            value = float(hit.get('score') or 0.0) - penalty * repeated
            if best_value is None or value > best_value:
                best_index, best_value = index, value
        picked = remaining.pop(best_index)
        chosen.append(picked)
        used[key_of(picked)] = used.get(key_of(picked), 0) + 1

    return chosen


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    limit: int,
    extra: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    """按名次融合多路结果。

    主路是向量和关键词；`extra` 里是**查询拆分**产生的子查询召回
    （"证券 原问题"、"期货 原问题"各一路）。

    子查询的命名带上业务线（`vector·证券`），这是刻意的：
    结果里会标出"这一条是拆出来那一路召回的"，出问题时能一眼看出
    是哪条子查询把无关内容带进来的。
    """

    merged: dict[str, dict[str, Any]] = {}

    sources: list[tuple[str, list[dict[str, Any]]]] = [
        ('vector', vector_hits),
        ('bm25', bm25_hits),
    ]
    sources.extend(extra or [])

    for source, hits in sources:
        for rank, hit in enumerate(hits, start=1):
            key = str(hit.get('chunk_id') or f'{source}:{rank}')
            entry = merged.setdefault(
                key,
                {
                    **hit,
                    'fused_score': 0.0,
                    'retrieval_sources': [],
                    'rank_vector': None,
                    'rank_bm25': None,
                    'vector_score': None,
                    'bm25_score': None,
                },
            )
            entry['fused_score'] += 1.0 / (RRF_K + rank)
            if source not in entry['retrieval_sources']:
                entry['retrieval_sources'].append(source)

            # 用 startswith 而不是相等：子查询那几路叫 "vector·证券"，
            # 它们和主路一样是向量召回，分数该记在同一个字段上。
            if source.startswith('vector'):
                entry['rank_vector'] = rank
                entry['vector_score'] = hit.get('score')
            elif source.startswith('bm25'):
                entry['rank_bm25'] = rank
                entry['bm25_score'] = hit.get('score')

            # 补齐两路各自的字段（比如向量路没有 bm25_score）
            for field_name, value in hit.items():
                if field_name in {'retrieval_sources', 'retrieval_source'}:
                    continue
                if entry.get(field_name) in (None, '', []):
                    entry[field_name] = value

    ranked = sorted(merged.values(), key=lambda item: item['fused_score'], reverse=True)
    for position, entry in enumerate(ranked, start=1):
        entry['rank_fused'] = position
        if len(entry['retrieval_sources']) > 1:
            entry['retrieval_source'] = 'hybrid'
            entry['score'] = entry['fused_score']
        elif entry['retrieval_sources'] == ['bm25']:
            entry['retrieval_source'] = 'bm25'
            entry['score'] = entry['bm25_score']
        else:
            entry['retrieval_source'] = 'vector'
            entry['score'] = entry['vector_score']

    return ranked[:limit]
