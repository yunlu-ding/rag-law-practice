from __future__ import annotations

import logging
import time
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)

"""向量化客户端（阿里云百炼 / DashScope）。

为什么直接用官方 SDK，而不是用框架封装：
这一层的逻辑其实只有"分批调用 + 失败重试 + 错误说清楚"，
用 SDK 写大约三十行，每一行都能解释；
套一层框架封装反而要额外搞清楚它内部怎么分批、怎么重试。
**能自己写清楚的胶水，不值得为它引入一层抽象。**

为什么必须分批：
接口对单次请求的文本条数有上限，一次性提交几千条会直接报错。
分批还有一个副作用是好的——每批完成就能回调一次进度，
前端看到的进度条才是真实推进，而不是几个固定档位来回跳。
"""

# 单次请求最多提交多少条文本。
# 取一个保守值：小一点更稳（单批失败的重试代价低），代价是请求次数多一点。
DEFAULT_BATCH_SIZE = 10

# 失败重试次数。向量化是"贵且慢"的操作，遇到网络抖动就整份文档失败太可惜。
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


class EmbeddingError(RuntimeError):
    """向量化失败。

    单独定义一个异常类型，是为了让上层能把"模型没配好"和"代码写错了"
    区分开——这两种情况给用户的提示完全不同。
    """


def _require_api_key() -> str:
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise EmbeddingError(
            '还没有配置百炼 API Key。请在 backend/.env 里填 DASHSCOPE_API_KEY。'
            '向量化、重排、生成都用它。'
        )
    return settings.dashscope_api_key


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback=None,
) -> list[list[float]]:
    """把一批文本转成向量，返回顺序与输入严格一致。

    顺序一致这一点必须保证：向量和切片是靠**下标**对应起来的，
    一旦顺序错了，向量就会挂到别的切片上——而这种错误不会报错，
    只会表现为"检索结果莫名其妙"，极难排查。
    所以下面按 text_index 显式排序，而不是相信返回顺序。
    """

    if not texts:
        return []

    api_key = _require_api_key()
    settings = get_settings()
    model = settings.embedding_model

    import dashscope

    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        response = _call_with_retry(
            dashscope=dashscope,
            model=model,
            batch=batch,
            api_key=api_key,
        )
        embeddings = (response.output or {}).get('embeddings') or []
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f'向量化返回条数与请求不一致：请求 {len(batch)} 条，返回 {len(embeddings)} 条'
            )
        ordered = sorted(embeddings, key=lambda item: item.get('text_index', 0))
        vectors.extend([list(item['embedding']) for item in ordered])

        if progress_callback is not None:
            progress_callback(start + len(batch), total)

    return vectors


def embed_query(text: str) -> list[float]:
    """把用户问题转成向量。"""

    vectors = embed_texts([text])
    if not vectors:
        raise EmbeddingError('向量化返回为空')
    return vectors[0]


def _call_with_retry(*, dashscope, model: str, batch: list[str], api_key: str):
    """调用接口，失败按退避策略重试。

    重试只针对"可能自己会好"的失败——网络抖动、限流。
    参数错误、额度用尽这类重试也没用的，直接抛出并附上原文，
    省得用户等三轮重试之后才看到一个含糊的错误。
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = dashscope.TextEmbedding.call(
                model=model,
                input=batch,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning('[EMBED] 第 %s 次调用异常: %s', attempt, exc)
        else:
            if getattr(response, 'status_code', None) == 200:
                return response
            last_error = EmbeddingError(
                f'百炼返回 {getattr(response, "status_code", "?")}: '
                f'{getattr(response, "code", "")} {getattr(response, "message", "")}'
            )
            logger.warning('[EMBED] 第 %s 次调用被拒: %s', attempt, last_error)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise EmbeddingError(f'向量化连续 {MAX_RETRIES} 次失败：{last_error}')


@lru_cache(maxsize=1)
def embedding_dimension() -> int:
    """配置里声明的向量维度。

    它必须和模型的真实输出维度一致，否则写入向量库会失败。
    与其在第一次写入时才报错，不如在配置校验阶段就撞出来——
    这也是为什么向量库的维度是从配置读的，而不是猜的。
    """

    return get_settings().milvus_dimension
