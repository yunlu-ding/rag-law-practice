from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.api.v1.system import build_health
from app.config import get_settings
from app.core.postgres import DatabaseNotConfiguredError, init_database
from app.schemas.common import HealthResponse
from app.services.document_service import resume_interrupted_documents
from app.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期。

    当前阶段只做轻量初始化，**不强制探测远端依赖**：
    某个外部服务暂时不可用，不应该让服务本身起不来。
    想确认依赖状态，去 /api/v1/system/health 看，而不是靠启动失败来发现。
    """

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info('上传目录已就绪: %s', upload_dir)

    # 建表。数据库没配好时不阻断启动——
    # 让服务起来、由前端明确告诉你"PostgreSQL 还没配"，
    # 比服务直接起不来更容易定位。
    try:
        init_database()
        logger.info('数据表已就绪')
    except DatabaseNotConfiguredError as exc:
        logger.warning('数据库未配置，文档相关接口将不可用：%s', exc)
    except Exception as exc:  # noqa: BLE001
        logger.error('建表失败，文档相关接口可能不可用：%s', exc)

    # 把上次没处理完的文档重新捡起来。
    # 后台任务跑在进程内存里，服务一重启任务就丢了；
    # 没有这一步，那些文档会永远停在"正在解析"，用户看着一个不动的进度条。
    try:
        resumed = resume_interrupted_documents()
        if resumed:
            logger.info('已恢复 %s 份未处理完的文档', resumed)
    except Exception:  # noqa: BLE001
        logger.exception('恢复中断文档失败')

    logger.info('应用启动: %s (%s) | %s', settings.app_name, settings.app_env, settings.build_stage)
    yield
    logger.info('应用停止: %s', settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description=(
        '面向证券期货监管法规的知识库问答系统。\n\n'
        '目标不是"能搜到资料"，而是**判断可核对**：\n'
        '- 每条回答都能指到原文，并注明依据的效力层级；\n'
        '- 知识库里没有依据时，明确说没有，不编。'
    ),
    openapi_url=f'{settings.api_v1_prefix}/openapi.json',
    docs_url=f'{settings.api_v1_prefix}/docs',
    redoc_url=f'{settings.api_v1_prefix}/redoc',
    lifespan=lifespan,
)

# 开发期前端如果单独跑 dev server，需要放行这个来源；
# 正式部署时前端由本服务同源托管，跨域配置实际上用不到。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get('/health', response_model=HealthResponse, tags=['system'])
def root_health() -> HealthResponse:
    """根路径健康检查，方便用 curl 或监控直接探活。"""

    return build_health(settings)


# 前端静态文件挂载在最后：这样 /api/... 会先被上面的路由匹配到，
# 剩下的请求才落到静态文件。
web_dir = Path(settings.web_dir)
web_dir.mkdir(parents=True, exist_ok=True)
app.mount('/', StaticFiles(directory=str(web_dir), html=True), name='web')
