from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import date

from sqlalchemy import select

from app.core.postgres import get_session_factory
from app.models.usage import UsageDaily

logger = logging.getLogger(__name__)

"""限额与熔断。

**为什么这一层是上线必需项，不是可选项：**
公网地址 + 你自己的 API Key = 任何人点一下都在花你的钱。
而且公网上的 AI 接口会被扫描器盯上，被拿去刷额度是常见事故，
不是理论风险。

所以只要联网，下面四道闸缺一不可：

| 闸门 | 挡什么 |
|---|---|
| 单 IP 每分钟限频 | 脚本批量刷 |
| 单 IP 每日问答上限 | 一个人聊一整天 |
| 全局每日问答上限 | 流量突然变大（最后一道闸） |
| 熔断 | 超限之后明确拒绝，而不是继续烧钱 |

**为什么用"次数"封顶，而不是"金额"封顶：**
金额需要实时价格数据，而价格会调整——写死在代码里的单价迟早会过期，
过期之后"预算封顶"就变成了一个假的安全感。
次数是等价的控制手段，而且不依赖任何外部数据。
token 用量照样记录，需要换算成本时用真实账单一除即可。
"""


class RateLimiter:
    """内存滑动窗口限流。

    为什么用内存而不是 Redis：
    我们是单实例部署，内存计数就够。引入 Redis 是为一个不存在的问题付运维成本。
    **代价要写清楚**：进程重启后计数清零；将来要是多实例部署，
    这个限流会失效（每个实例各算各的），那时候必须换成集中的计数。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        """记一次访问并判断是否超限，返回 (是否放行, 剩余次数)。"""

        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False, 0
            bucket.append(now)
            return True, limit - len(bucket)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter()
    return _limiter


class DailyCounter:
    """按 key 统计"当日"次数（内存）。

    为什么不用数据库：这是高频写入，而且丢失的后果很轻（重启后计数清零，
    最坏情况是当天多放行一些请求）。**为了这种精度去引入一张高频写的表不划算。**

    跨天自动归零：日期一变就把计数清空，避免"昨天的用量算到今天头上"。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = date.today()
        self._counts: dict[str, int] = {}

    def incr(self, key: str) -> int:
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day = today
                self._counts.clear()
            self._counts[key] = self._counts.get(key, 0) + 1
            return self._counts[key]

    def peek(self, key: str) -> int:
        with self._lock:
            if date.today() != self._day:
                return 0
            return self._counts.get(key, 0)


_daily: DailyCounter | None = None


def get_daily_counter() -> DailyCounter:
    global _daily
    if _daily is None:
        _daily = DailyCounter()
    return _daily


class LimitDecision:
    """一次限额判断的结果。"""

    def __init__(
        self,
        allowed: bool,
        message: str = '',
        remaining: int | None = None,
        status_code: int = 429,
    ) -> None:
        self.allowed = allowed
        self.message = message
        self.remaining = remaining
        # 429 = 你太快了，等一下还能用；503 = 额度用完了，今天不行了。
        # 这两个状态码含义不同，前端可以据此给出不同的提示。
        self.status_code = status_code


def check_limits(*, kind: str, client_ip: str) -> LimitDecision:
    """四道闸依次判断。任何一道没过就返回明确的拒绝理由。

    **拒绝信息必须说清楚是哪一道闸、还剩多少**，
    而不是笼统的"请求过于频繁"——
    用户看到"你不小心点太快了，等一分钟"和看到"系统繁忙"，
    反应是完全不同的。
    """

    from app.config import get_settings

    settings = get_settings()

    # 第 1 道：每分钟限频（所有接口共用）
    allowed, remaining = get_rate_limiter().check(
        f'minute:{client_ip}',
        limit=settings.rate_limit_per_ip_per_min,
        window_seconds=60,
    )
    if not allowed:
        record_blocked(reason=f'分钟限频 ip={client_ip}')
        return LimitDecision(
            False,
            f'操作太快了：每分钟最多 {settings.rate_limit_per_ip_per_min} 次，请稍等一会儿再试。',
            status_code=429,
        )

    if kind != 'qa':
        return LimitDecision(True, remaining=remaining)

    # 第 2 道：单 IP 每日问答上限
    ip_count = get_daily_counter().incr(f'qa:{client_ip}')
    if settings.daily_qa_limit_per_ip and ip_count > settings.daily_qa_limit_per_ip:
        record_blocked(reason=f'单IP每日上限 ip={client_ip}')
        return LimitDecision(
            False,
            f'今天你这边已经问了 {settings.daily_qa_limit_per_ip} 次，达到单日上限。明天再来。',
            status_code=503,
        )

    # 第 3 道：全局每日问答上限（最后一道闸）
    usage = peek_today()
    if settings.daily_qa_limit_global and usage['qa_count'] >= settings.daily_qa_limit_global:
        record_blocked(reason='全局每日上限')
        return LimitDecision(
            False,
            '今天整个演示环境的问答额度已经用完了。'
            '这是为了防止 API 额度被意外刷完而设的闸门，明天会自动恢复。',
            status_code=503,
        )

    return LimitDecision(True, remaining=remaining)


def _today() -> date:
    return date.today()


def get_or_create_today(db) -> UsageDaily:
    """取今天的用量记录，没有就建一条。"""

    today = _today()
    record = db.execute(
        select(UsageDaily).where(UsageDaily.usage_date == today)
    ).scalar_one_or_none()
    if record is None:
        record = UsageDaily(usage_date=today)
        db.add(record)
        db.commit()
        db.refresh(record)
    return record


def peek_today() -> dict:
    """读今天的用量（只读，不建记录）。读不到时返回 0，不抛异常——
    用量展示失败不该影响任何主流程。"""

    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            record = db.execute(
                select(UsageDaily).where(UsageDaily.usage_date == _today())
            ).scalar_one_or_none()
            if record is None:
                return {'qa_count': 0, 'retrieval_count': 0, 'blocked_count': 0,
                        'prompt_tokens': 0, 'completion_tokens': 0}
            return {
                'qa_count': record.qa_count,
                'retrieval_count': record.retrieval_count,
                'blocked_count': record.blocked_count,
                'prompt_tokens': record.prompt_tokens,
                'completion_tokens': record.completion_tokens,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning('[USAGE] 读取当日用量失败（不影响主流程）: %s', exc)
        return {'qa_count': 0, 'retrieval_count': 0, 'blocked_count': 0,
                'prompt_tokens': 0, 'completion_tokens': 0}


def record_usage(*, kind: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """记一次用量。

    失败只记日志，不抛异常：计费统计不准是可接受的，
    但"因为统计失败而让用户问不了问题"不可接受。
    """

    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            record = get_or_create_today(db)
            if kind == 'qa':
                record.qa_count += 1
            elif kind == 'retrieval':
                record.retrieval_count += 1
            record.prompt_tokens += prompt_tokens
            record.completion_tokens += completion_tokens
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning('[USAGE] 记录用量失败（不影响主流程）: %s', exc)


def record_blocked(*, reason: str) -> None:
    """记一次被拦截。被拦截的次数要能看到——
    "限额经常触发"和"限额从来没触发过"是两个完全不同的信号。"""

    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            record = get_or_create_today(db)
            record.blocked_count += 1
            db.commit()
        logger.warning('[LIMIT] 已拦截一次请求: reason=%s', reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning('[USAGE] 记录拦截失败: %s', exc)
