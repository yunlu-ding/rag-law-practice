from __future__ import annotations

from fastapi import APIRouter

from app.config import Settings, get_settings
from app.schemas.common import ConfigResponse, DependencyItem, HealthResponse, KnobItem

router = APIRouter(prefix='/system', tags=['system'])


def _is_configured(value: str | None) -> bool:
    return bool(value and str(value).strip())


def build_dependency_status(settings: Settings) -> list[DependencyItem]:
    """汇总各外部依赖的配置状态。

    这是"可观测"原则的第一步：**先能看见自己缺什么**。
    搭建过程中最浪费时间的情形，是不知道卡在哪一环；
    把这几个状态显式列出来，一眼就能定位。
    """

    return [
        DependencyItem(
            name='dashscope',
            label='百炼（对话 + 向量化 + 重排）',
            configured=_is_configured(settings.dashscope_api_key),
            required_now=True,
            active_from='阶段三',
            note='向量化只能用这家：DeepSeek 没有 embedding 接口',
        ),
        DependencyItem(
            name='deepseek',
            label='DeepSeek（备用生成模型）',
            configured=_is_configured(settings.deepseek_api_key),
            required_now=False,
            active_from='阶段三',
            note='用于跑"同一批检索结果，两个模型谁判定更准"的对照实验',
        ),
        DependencyItem(
            name='postgres',
            label='PostgreSQL（文档 / 切片 / 日志）',
            configured=_is_configured(settings.postgres_dsn),
            required_now=True,
            active_from='阶段二',
            note='主数据在这里；与向量库不一致时以它为准',
        ),
        DependencyItem(
            name='milvus',
            label='Milvus / Zilliz Cloud（向量索引）',
            configured=_is_configured(settings.milvus_uri),
            required_now=True,
            active_from='阶段三',
            note='只存向量与检索副本，可随时重建',
        ),
    ]


def build_health(settings: Settings) -> HealthResponse:
    dependencies = build_dependency_status(settings)

    # 当前阶段（骨架）不强制依赖任何外部服务，所以 ok 只看"必需项是否就绪"。
    # 这一点与机构项目不同：那边 Redis 不通会让整体判定失败，
    # 但业务链路其实不依赖 Redis——"没用到却影响可用性判断"是典型的配置债。
    ok = all(item.configured for item in dependencies if item.required_now)

    return HealthResponse(
        ok=ok,
        app_name=settings.app_name,
        app_version=settings.app_version,
        app_env=settings.app_env,
        build_stage=settings.build_stage,
        dependencies=dependencies,
    )


@router.get('/health', response_model=HealthResponse)
def system_health() -> HealthResponse:
    """服务与依赖状态。"""

    return build_health(get_settings())


@router.get('/config', response_model=ConfigResponse)
def system_config() -> ConfigResponse:
    """当前生效的可调参数。

    为什么做成接口而不是只写在文档里：
    **参数是拿来拧的，而拧之前必须先知道现在是多少。**
    之后每次做对照实验，改的也就是这张表里的东西。
    """

    settings = get_settings()
    knobs = [
        KnobItem(
            group='检索',
            name='RETRIEVAL_TOP_K',
            value=str(settings.retrieval_top_k),
            note='喂给模型的证据条数：多了有噪音，少了会漏',
        ),
        KnobItem(
            group='检索',
            name='RERANK_ENABLED',
            value=str(settings.rerank_enabled),
            note='关掉可以做"有重排 / 没重排"的对照实验',
        ),
        KnobItem(
            group='检索',
            name='RERANK_CANDIDATE_K',
            value=str(settings.rerank_candidate_k),
            note='精排前先粗筛多少条；实测瓶颈在排序不在召回，所以池子可以开大',
        ),
        KnobItem(
            group='检索',
            name='RERANK_MODEL',
            value=settings.rerank_model,
            note='重排模型；调用失败会自动退回原顺序，不会让问答挂掉',
        ),
        KnobItem(
            group='生成',
            name='GENERATION_PROVIDER',
            value=settings.generation_provider,
            note='tongyi / deepseek 可切换，用于对照实验',
        ),
        KnobItem(
            group='生成',
            name='MODEL',
            value=settings.model,
            note='对话模型名',
        ),
        KnobItem(
            group='生成',
            name='REFUSE_SCORE_THRESHOLD',
            value=str(settings.refuse_score_threshold),
            note='低于该分数直接拒答；要靠评测校准，不能凭感觉填',
        ),
        KnobItem(
            group='向量',
            name='EMBEDDING_MODEL',
            value=settings.embedding_model,
            note='改动等于重建整个向量库，上线前定死',
        ),
        KnobItem(
            group='向量',
            name='MILVUS_DIMENSION',
            value=str(settings.milvus_dimension),
            note='必须与 embedding 模型的输出维度一致',
        ),
        KnobItem(
            group='上传',
            name='MAX_UPLOAD_SIZE_MB',
            value=str(settings.max_upload_size_mb),
            note='教材类 PDF 体积较大，默认给到 100MB',
        ),
        KnobItem(
            group='限额',
            name='RATE_LIMIT_PER_IP_PER_MIN',
            value=str(settings.rate_limit_per_ip_per_min),
            note='上线后防刷的第一道闸',
        ),
        KnobItem(
            group='限额',
            name='DAILY_BUDGET_CNY',
            value=str(settings.daily_budget_cny),
            note='最后一道闸：超过就拒绝服务，而不是继续烧钱',
        ),
    ]
    return ConfigResponse(build_stage=settings.build_stage, knobs=knobs)
