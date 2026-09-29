from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.chunks import router as chunks_router
from app.api.v1.documents import router as documents_router
from app.api.v1.eval import router as eval_router
from app.api.v1.qa import router as qa_router
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.system import router as system_router
from app.api.v1.stats import router as stats_router
from app.api.v1.usage import router as usage_router

api_router = APIRouter()

# 所有 v1 接口统一在这里汇总。
api_router.include_router(system_router)
api_router.include_router(documents_router)
api_router.include_router(chunks_router)
api_router.include_router(retrieval_router)
api_router.include_router(qa_router)
api_router.include_router(usage_router)
api_router.include_router(stats_router)
api_router.include_router(eval_router)
