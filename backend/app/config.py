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

    # 结果多样性：同一份文件被重复选中时，每条往下压多少分。
    #
    # 为什么需要它：重排是按"这一段有多像问题"排序的，而**同一份文件里
    # 相邻的切片内容高度重叠**，于是一份文件很容易一次占满前 5 个位置。
    # 实测基线里 28 个失败题有 14 个都是这种情况——排头的是《适当性管理办法》
    # 问答，它用口语解释法规，语气和用户提问最接近，
    # 把真正作为依据的法规条文挤了出去。
    #
    # 代价是"单文档问题"可能被换掉一两个也相关的邻居。所以只对**第 3 名之后**
    # 做多样性处理（前两名是最强证据，不动）——重复的边际信息量本来就低。
    #
    # 设为 0 就退回原行为。
    rerank_diversity_penalty: float = Field(
        default=0.08, alias='RERANK_DIVERSITY_PENALTY'
    )
    # 前几名不参与多样性打散。
    rerank_diversity_protect: int = Field(default=2, alias='RERANK_DIVERSITY_PROTECT')

    # 查询拆分：把"证券和期货有什么区别"这类对比问题拆成按业务线的子查询。
    #
    # ⚠️ **默认关闭**，因为实测它没有改善（77% → 77%）。
    #
    # 原来推断"那类题的第二个业务线压根不在候选池里"，实测发现推断错了：
    # 子查询确实把另一条业务线的候选带进来了，但**重排用它不上去**——
    # 排头的仍然是《办法》问答里两段泛泛而谈的段落，
    # 而《证券法》第八十八条、《期货和衍生品法》第五十条的重排分只有 0.15 上下。
    #
    # 所以根因不是"候选池缺文件"，而是**重排对"对比类"问题偏好概述性段落、
    # 压过具体条文**。扩候选解决不了它。
    #
    # 代码保留，是因为它是"按业务线保底"那个改法的前提（先得把候选召进来，
    # 才谈得上给每条业务线留位置）。等两者一起测出效果，再把它打开。
    query_split_enabled: bool = Field(default=False, alias='QUERY_SPLIT_ENABLED')
    # 每条业务线保底位置的分数下限。低于它就说明"这条线里也没有像样的东西"，
    # 与其硬塞一个不相干的切片，不如让位给别的结果。
    query_split_min_score: float = Field(default=0.05, alias='QUERY_SPLIT_MIN_SCORE')

    # Wiki 词条通道。开着时，对比类问题会先尝试匹配词条。
    #
    # 对照实验用得到：关掉它就是"纯 RAG"，打开就是"RAG + Wiki"。
    wiki_enabled: bool = Field(default=True, alias='WIKI_ENABLED')

    # ------------------------------
    # 知识边界（阶段三启用）
    #
    # ⚠️ 这个阈值**只管一条路**：混合检索（向量 + 关键词 + 重排）的结果。
    #
    # 它原来管全部，后来被拆开了，因为"有没有"和"像不像"是两类问题：
    #
    #   · **条款直查**（问题里指名了法规名 + 条号）→ 确定性判断，不看分数。
    #     查得到就答，查不到就是"库里没有这一条"。
    #     分数在这里根本不存在（关系库取出来的东西没有相似度），
    #     拿阈值去比它，等于把一个确定的事实说成一个猜测。
    #   · **词条命中** → 同样不走阈值。词条是已编译的跨文档结论，
    #     它命中本身就是"有依据"，不存在"像不像"。
    #   · 只有**混合检索**这条路，结果是"最像的几段"，
    #     "够不够像"才是一个需要标定的问题。
    #
    # 而且阈值只对**重排分**有效。重排不可用时（关闭、或调用失败退回原顺序），
    # `score` 会变成融合分甚至 BM25 分（量纲差两个数量级），
    # 那时系统进入降级模式并在答案里标注，**不拿别的量纲硬套阈值**。
    # 理由很直接：欠费那天所有查询的 score 都是 47 上下，
    # 阈值 0.5 会让"系统最坏的一天"恰好变成"闸门最松的一天"。
    #
    # 阈值怎么定：**从分布里量出来，不用评测通过率反解**（工具/标定拒答阈值.py）。
    #
    # 现取值 0.13，判据是**零误伤**：
    #
    #   评测里 87 道判"通过"的题，最高重排分最低的那一道是 **0.140**
    #   （D3-15"冷静期是什么意思"）。阈值取不超过它，就**不会拒掉任何一道
    #   本来能答对的题**。取 0.13 留了一点余量，同时把最离谱的几条挡住：
    #
    #       0.094  明天上海天气怎么样
    #       0.084  怎么做红烧肉
    #       0.129  帮我写一首关于春天的诗
    #
    # ⚠️ 这个数字**小得反常，而它是有原因的**——不要以为调大一点会更好：
    #
    #   探针里 11 条"该拒"的题，分数从 0.084 到 0.557；而"该答"的题从 0.196 到 0.748。
    #   **两类分布是重叠的**。具体说：
    #       "香港证监会对专业投资者怎么界定"（该拒） 0.293
    #       "基金募集机构投资者适当性管理实施指引什么时候实施"（该答） 0.196
    #   一个该拒的题，分数比一个该答的题**还高**。
    #
    #   所以阈值取大一点，收益是"多拒几道该拒的"，代价是"开始拒掉答对的题"：
    #       0.18 → 拒对 6/11，但同时拒掉 5 道本来答对的
    #       0.30 → 拒对 9/11，但同时拒掉 21 道本来答对的
    #   这不是调参能解决的，是**单靠一个分数分不开这两类**。
    #
    #   剩下的该拒情况交给**确定性判据**（见 app/rag/refusal.py）：
    #   条号不存在 → citation_missing；境外法域、生成类任务 → 需要各自的判据，
    #   目前还没有做，是已知缺口。
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
