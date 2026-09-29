from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

"""全局配置。

设计目标（沿用机构项目的三条，因为这三条是对的）：
1. 所有外部依赖配置集中在这里，不散落到业务代码；
2. 配置项名与 .env 里的名字一一对应，便于排查；
3. 启动阶段就完成校验。

与机构项目的一处**重要差别**：
    机构项目把 DASHSCOPE_API_KEY / MILVUS_COLLECTION / POSTGRES_DSN / REDIS_URL
    设为必填，缺任何一个服务都起不来。
    本项目把它们全部设为**可选**，理由是我们会分阶段搭建：
    还没接数据库的时候，服务也应该能起来、前端也应该能看到"哪一项还没配"。
    把一个"还没配"的依赖变成"服务起不来"，会让整个搭建过程变得很难受。
"""

# backend/ 目录：config.py 位于 backend/app/ 下，上两级就是 backend/
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / '.env'


class Settings(BaseSettings):
    """应用配置。字段名与 .env 变量名一一对应（不区分大小写）。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # ------------------------------
    # 应用本身
    # ------------------------------
    app_name: str = Field(default='金融监管法规知识库', alias='APP_NAME')
    app_version: str = '0.1.0'
    app_env: str = Field(default='development', alias='APP_ENV')
    debug: bool = Field(default=False, alias='DEBUG')
    api_v1_prefix: str = '/api/v1'
    allowed_origins: list[str] = Field(
        default_factory=lambda: ['http://localhost:5173'],
        alias='ALLOWED_ORIGINS',
    )

    # 当前实施阶段。写进接口返回、由前端直接展示，随时能知道做到哪一步。
    build_stage: str = '阶段八 · 限额与上线准备'

    # ------------------------------
    # 存储
    # ------------------------------
    storage_root: str = Field(default='storage', alias='STORAGE_ROOT')
    upload_dir_name: str = Field(default='uploads', alias='UPLOAD_DIR_NAME')
    max_upload_size_mb: int = Field(default=100, alias='MAX_UPLOAD_SIZE_MB')

    # ------------------------------
    # 大模型（阶段三启用；现在允许为空）
    # ------------------------------
    dashscope_api_key: str | None = Field(default=None, alias='DASHSCOPE_API_KEY')
    deepseek_api_key: str | None = Field(default=None, alias='DEEPSEEK_API_KEY')
    model: str = Field(default='qwen-plus', alias='MODEL')
    embedding_model: str = Field(default='text-embedding-v1', alias='EMBEDDING_MODEL')

    # 生成模型来源：tongyi / deepseek。
    # 做成配置项不是为了"留个后路"，而是为了能跑对照实验——
    # "同一批检索结果，两个模型谁判定更准"这个问题只能靠数字回答。
    generation_provider: str = Field(default='tongyi', alias='GENERATION_PROVIDER')

    # ------------------------------
    # 向量库（阶段三启用）
    # ------------------------------
    milvus_uri: str | None = Field(default=None, alias='MILVUS_URI')
    milvus_token: str | None = Field(default=None, alias='MILVUS_TOKEN')
    milvus_collection: str = Field(default='vibe_regulation_knowledge', alias='MILVUS_COLLECTION')
    milvus_dimension: int = Field(default=1536, alias='MILVUS_DIMENSION')
    milvus_host: str = Field(default='127.0.0.1', alias='MILVUS_HOST')
    milvus_port: int = Field(default=19530, alias='MILVUS_PORT')

    # ------------------------------
    # 关系库（阶段二启用）
    # ------------------------------
    postgres_dsn: str | None = Field(default=None, alias='POSTGRES_DSN')

    # ------------------------------
    # 检索参数（阶段三起生效）
    #
    # 这一组是后续所有对照实验的操作面，所以全部做成配置项：
    # 调参不需要改代码，只需要改 .env 再重启。
    # ------------------------------
    retrieval_top_k: int = Field(default=5, alias='RETRIEVAL_TOP_K')
    rerank_enabled: bool = Field(default=True, alias='RERANK_ENABLED')
    rerank_model: str = Field(default='gte-rerank-v2', alias='RERANK_MODEL')
    rerank_candidate_k: int = Field(default=100, alias='RERANK_CANDIDATE_K')

    # ------------------------------
    # 知识边界（阶段三启用）
    #
    # 检索最高分低于阈值就直接拒答，不进生成。
    # 这是硬闸门；提示词只是第二道防线——提示词是软约束，拦不住模型在最需要拒答时仍然作答。
    # ------------------------------
    refuse_score_threshold: float = Field(default=0.0, alias='REFUSE_SCORE_THRESHOLD')

    # ------------------------------
    # 演示限额（阶段四启用）
    #
    # 公网 URL + 自有 API Key = 任何人点击都在花钱，所以这几项是上线必需，不是可选项。
    # ------------------------------
    rate_limit_per_ip_per_min: int = Field(default=20, alias='RATE_LIMIT_PER_IP_PER_MIN')
    session_turn_limit: int = Field(default=30, alias='SESSION_TURN_LIMIT')
    daily_budget_cny: float = Field(default=10.0, alias='DAILY_BUDGET_CNY')
    # 按次数封顶（而不是按金额）：金额需要单价，单价会变，
    # 写死的单价过期之后"预算封顶"就变成了假的安全感。次数是等价且不会失效的控制手段。
    daily_qa_limit_per_ip: int = Field(default=30, alias='DAILY_QA_LIMIT_PER_IP')
    daily_qa_limit_global: int = Field(default=300, alias='DAILY_QA_LIMIT_GLOBAL')
    # 演示模式：开启后，真机问答需要口令。
    # 面试时你自己用真机、把预置示例给面试官看，可以用它进一步压低风险。
    demo_mode: bool = Field(default=False, alias='DEMO_MODE')
    demo_passcode: str | None = Field(default=None, alias='DEMO_PASSCODE')
    # 限额总开关。本地开发可以关掉，**上线前必须确认它是开的**（部署检查清单里有这一条）。
    limits_enabled: bool = Field(default=True, alias='LIMITS_ENABLED')

    @field_validator('allowed_origins', mode='before')
    @classmethod
    def normalize_allowed_origins(cls, value: Any) -> list[str]:
        """兼容三种写法：JSON 数组、逗号分隔字符串、Python 列表。

        这样 .env 里怎么写都不会因为格式问题起不来。
        """

        if value is None:
            return ['http://localhost:5173']

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith('['):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError('ALLOWED_ORIGINS JSON 必须是数组')
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(',') if item.strip()]

        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]

        raise TypeError('ALLOWED_ORIGINS 必须是列表或逗号分隔字符串')

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upload_dir(self) -> str:
        """上传文件的绝对路径。"""

        return str(BASE_DIR / self.storage_root / self.upload_dir_name)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def web_dir(self) -> str:
        """前端静态文件目录。

        前端由后端同源托管：一个进程、一个端口、没有跨域、没有构建步骤。
        这是"面试官点开链接就能用"成本最低的实现方式。
        """

        return str(BASE_DIR.parent / 'web')

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_milvus_uri(self) -> str:
        """统一返回 Milvus 连接地址，其它模块只依赖这一个字段。"""

        if self.milvus_uri:
            return self.milvus_uri
        return f'http://{self.milvus_host}:{self.milvus_port}'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eval_dir(self) -> str:
        """评测目录。

        评测结果目前是 CSV 文件（评测脚本产出），而不是数据库表。
        这么设计是刻意的：**评测结果是"一次性判决"，不是业务数据**——
        它跟着代码版本走，应该能直接打开看、能提交进仓库、能随代码一起回滚。
        为了工作台能展示它，这里给出目录位置，由接口层去读。
        """

        return str(BASE_DIR.parent / '评测')

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回配置单例。

    做缓存是为了保证整个进程内读取配置的行为一致，
    同时避免在高频依赖注入场景下反复解析 .env。
    """

    return Settings()
