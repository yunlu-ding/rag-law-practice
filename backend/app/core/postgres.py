from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


class DatabaseNotConfiguredError(RuntimeError):
    """数据库还没配置。

    单独定义一个异常，是为了让上层能把它翻译成"明确告诉用户缺什么"的提示，
    而不是抛一个看不懂的连接错误。
    """


def _normalize_dsn(dsn: str) -> str:
    """把通用 DSN 转成 SQLAlchemy 推荐的方言写法。

    用户在 .env 里通常写 `postgresql://...`，
    但项目用的是 psycopg 3 驱动，显式写成 `postgresql+psycopg://` 更清晰，
    也避免 SQLAlchemy 去猜用哪个驱动。
    """

    if dsn.startswith('postgresql://'):
        return 'postgresql+psycopg://' + dsn[len('postgresql://'):]
    return dsn


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """SQLAlchemy Engine 单例。

    pool_pre_ping=True：从连接池取出连接前先做一次轻量探测，
    避免长时间闲置后拿到一个已经失效的连接（这个坑在本地开发时很常见，
    因为数据库经常被单独关掉再打开）。
    """

    settings = get_settings()
    if not settings.postgres_dsn:
        raise DatabaseNotConfiguredError(
            '还没有配置 PostgreSQL。请在 backend/.env 里填 POSTGRES_DSN，'
            '例如 postgresql://postgres@127.0.0.1:5432/vibe_rag'
        )
    return create_engine(_normalize_dsn(settings.postgres_dsn), pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Session 工厂单例。

    autoflush=False：显式提交前不做隐式刷盘，少一些意外的 SQL。
    expire_on_commit=False：提交后对象还能继续访问，方便 service 层直接用返回值。
    """

    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def get_db() -> Generator[Session, None, None]:
    """每个请求一个独立 Session，请求结束统一关闭。"""

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_database() -> None:
    """建表。

    ⚠️ 已知边界：`create_all` 只会**创建不存在的表**，
    不会给已经存在的表加字段。所以以后凡是改模型字段，
    都必须另外执行一次 ALTER TABLE（或者引入迁移工具），
    否则会出现"代码里明明加了字段，数据库里却没有"的诡异现象。
    """

    from app.models import Base

    Base.metadata.create_all(bind=get_engine())


def reset_engine_cache() -> None:
    """清掉 Engine 缓存。

    配置变更（比如改了 DSN）之后需要它，否则会一直用旧连接。
    目前主要给自检脚本用。
    """

    get_engine.cache_clear()
    get_session_factory.cache_clear()
