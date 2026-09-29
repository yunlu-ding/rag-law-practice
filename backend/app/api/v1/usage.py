from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.limits import get_daily_counter, peek_today

router = APIRouter(prefix='/usage', tags=['usage'])


class UsageResponse(BaseModel):
    usage_date: str
    qa_count: int = Field(description='今日问答次数')
    retrieval_count: int = Field(description='今日独立检索次数')
    blocked_count: int = Field(description='今日被限额拦截的次数')
    prompt_tokens: int = 0
    completion_tokens: int = 0
    limits_enabled: bool = Field(description='限额是否生效——上线前必须确认它是 true')
    daily_qa_limit_global: int
    daily_qa_limit_per_ip: int
    rate_limit_per_ip_per_min: int
    your_ip_qa_count: int = Field(description='你自己今天问了多少次（用于自查）')
    remaining_global: int = Field(description='全局还剩多少次问答额度')


@router.get('', response_model=UsageResponse)
def get_usage(request: Request) -> UsageResponse:
    """当前用量与限额状态。

    这个接口有两个用途：
    1. 给前端显示"还剩多少额度"，让用户知道为什么突然被拦；
    2. **给上线前的自查用**——`limits_enabled` 会明确告诉你限额到底有没有生效。
       很多"以为开了其实没开"的事故，都是因为没有一个地方能直接看出来。
    """

    from datetime import date

    settings = get_settings()
    usage = peek_today()
    remaining = max(settings.daily_qa_limit_global - usage['qa_count'], 0)

    client_ip = request.client.host if request.client else 'unknown'

    return UsageResponse(
        usage_date=date.today().isoformat(),
        qa_count=usage['qa_count'],
        retrieval_count=usage['retrieval_count'],
        blocked_count=usage['blocked_count'],
        prompt_tokens=usage['prompt_tokens'],
        completion_tokens=usage['completion_tokens'],
        limits_enabled=settings.limits_enabled,
        daily_qa_limit_global=settings.daily_qa_limit_global,
        daily_qa_limit_per_ip=settings.daily_qa_limit_per_ip,
        rate_limit_per_ip_per_min=settings.rate_limit_per_ip_per_min,
        your_ip_qa_count=get_daily_counter().peek(f'qa:{client_ip}'),
        remaining_global=remaining,
    )
