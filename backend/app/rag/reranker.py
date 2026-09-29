from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

"""检索结果重排。

为什么需要这一层，以及它是怎么被"实验"逼出来的：

最初的假设是"检索排不好，是因为候选池太小"。于是把候选池从 20 放大到 500，
结果是——池内召回率从七成涨到九成以上，**但 Top-5 命中率几乎不动**。

这个结果推翻了原假设：正确内容一直在池子里，只是排不到前面。
所以单纯调参收益接近零，必须换机制。

重排换的就是机制：向量检索把问题和文档**分别**编码再算相似度，
而重排模型把"问题 + 候选片段"**拼在一起**送进模型直接打分，
能建模两者之间的细粒度交互，精度高得多。代价是算得慢，
所以只能对粗筛出来的几十上百条做，不能对全库做。

典型流水线：召回几十上百条 → 重排 → 取前几条。

⚠️ 失败时的处理方式，和向量库删除刚好相反：
重排失败就**退回原有顺序**，让系统退化成"没有重排"的状态。
理由重排只影响结果的**排序质量**，不影响数据的**正确性**——
这是体验层的问题，降级优于中断。
"""


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """用重排模型重新排序，返回前 top_k 条。失败时原样返回前 top_k 条。"""

    settings = get_settings()
    if not settings.rerank_enabled or not hits:
        return hits[:top_k]

    documents = [str(hit.get('text') or '') for hit in hits]
    if not any(documents):
        return hits[:top_k]

    try:
        from dashscope import TextReRank

        response = TextReRank.call(
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
            return_documents=False,
            api_key=settings.dashscope_api_key,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning('[RERANK] 调用异常，退回原顺序: %s', exc)
        return hits[:top_k]

    if getattr(response, 'status_code', None) != 200:
        logger.warning(
            '[RERANK] 接口返回非 200，退回原顺序: code=%s message=%s',
            getattr(response, 'code', None),
            str(getattr(response, 'message', ''))[:200],
        )
        return hits[:top_k]

    results = getattr(getattr(response, 'output', None), 'results', None) or []
    if not results:
        logger.warning('[RERANK] 返回结果为空，退回原顺序')
        return hits[:top_k]

    reranked: list[dict[str, Any]] = []
    for item in results:
        index = item.get('index')
        if not isinstance(index, int) or not (0 <= index < len(hits)):
            continue
        hit = dict(hits[index])
        hit['rerank_score'] = item.get('relevance_score')
        # 记录重排前后的名次。没有这个数据，"重排到底有没有用"
        # 就只能靠感觉回答——而这是我们最不想靠感觉的地方。
        hit['rank_before_rerank'] = index + 1
        reranked.append(hit)

    if not reranked:
        return hits[:top_k]

    for position, hit in enumerate(reranked, start=1):
        hit['score'] = hit.get('rerank_score')
        hit['rank_after_rerank'] = position
        # 保持原来的召回来源不被覆盖，调试台里仍要能看出这一条从哪一路来
        hit['reranked'] = True

    logger.info(
        '[RERANK] 完成: candidates=%s returned=%s model=%s',
        len(documents),
        len(reranked),
        settings.rerank_model,
    )
    return reranked
