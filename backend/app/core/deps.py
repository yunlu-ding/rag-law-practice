from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.limits import check_limits
from app.core.postgres import DatabaseNotConfiguredError, get_db as _get_db


def get_database() -> Generator[Session, None, None]:
    """FastAPI 数据库依赖。

    把"数据库没配置"翻译成一句人话的 503，而不是让一个连接异常冒到前端。
    错误信息里直接写清楚该改哪个文件、填什么——
    **报错要能告诉人下一步做什么，而不只是说"出错了"。**
    """

    try:
        yield from _get_db()
    except DatabaseNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def enforce_limits(kind: str):
    """生成一个限额依赖。

    做成"按用途生成"而不是全局中间件，是因为**不同接口的额度不应该一样**：
    检索一次只花一点向量化的钱，问答一次要花检索加生成两份钱。
    用同一个额度管它们，要么放过问答、要么卡死检索。
    """

    def _guard(request: Request) -> None:
        settings = get_settings()
        if not settings.limits_enabled:
            return
        client_ip = request.client.host if request.client else 'unknown'
        decision = check_limits(kind=kind, client_ip=client_ip)
        if not decision.allowed:
            raise HTTPException(
                status_code=decision.status_code,
                detail=decision.message,
                headers={'Retry-After': '60'},
            )

    return _guard
