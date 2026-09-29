from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)

"""生成模型客户端（阿里云百炼 / 通义千问）。

为什么这一层薄得像一张纸：
它就是"发一次请求、把文本取回来"，没有什么值得抽象的。
业务上真正的判断（要不要拒答、引用怎么还原、结果怎么结构化）
都在 service 层，不在这里——
**把判断留在能被测试、能被解释的地方。**
"""

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


class GenerationError(RuntimeError):
    """生成失败。"""


@dataclass
class GenerationResult:
    """一次生成的结果。

    把 token 用量一起返回，是为了让"用量"这件事有真实数据可记。
    只返回文本的话，成本就只能靠估。
    """

    content: str
    total_tokens: int = 0


def generate(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> GenerationResult:
    """调用对话模型。"""

    settings = get_settings()
    if not settings.dashscope_api_key:
        raise GenerationError('还没有配置百炼 API Key（DASHSCOPE_API_KEY）')

    import dashscope

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()
        try:
            response = dashscope.Generation.call(
                model=settings.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                result_format='message',
                # 温度压低：这个场景要的是稳定复现，不是文采。
                # 判断题上"每次答案略有不同"是灾难——用户没法核对。
                temperature=temperature,
                api_key=settings.dashscope_api_key,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning('[LLM] 第 %s 次调用异常: %s', attempt, exc)
        else:
            if getattr(response, 'status_code', None) == 200:
                choices = getattr(getattr(response, 'output', None), 'choices', None) or []
                if not choices:
                    raise GenerationError('模型返回为空')
                content = choices[0].get('message', {}).get('content', '')
                usage = getattr(response, 'usage', None)
                logger.info(
                    '[LLM] 生成完成: model=%s 耗时=%sms tokens=%s',
                    settings.model,
                    int((time.perf_counter() - started) * 1000),
                    getattr(usage, 'total_tokens', None),
                )
                return GenerationResult(
                    content=str(content or ''),
                    total_tokens=int(getattr(usage, 'total_tokens', 0) or 0),
                )

            last_error = GenerationError(
                f'百炼返回 {getattr(response, "status_code", "?")}: '
                f'{getattr(response, "code", "")} {getattr(response, "message", "")}'
            )
            logger.warning('[LLM] 第 %s 次调用被拒: %s', attempt, last_error)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise GenerationError(f'生成连续 {MAX_RETRIES} 次失败：{last_error}')
