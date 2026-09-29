from __future__ import annotations

from pydantic import BaseModel, Field


class DependencyItem(BaseModel):
    """一个外部依赖的配置状态。

    这里刻意区分"已配置"和"当前阶段是否必需"：
    搭建是分阶段的，还没到那一步的依赖没配是正常的，
    不应该显示成错误——否则前端一直飘红，久了就没人看状态了。
    """

    name: str = Field(description='依赖标识')
    label: str = Field(description='中文说明')
    configured: bool = Field(description='是否已配置（有值且非空）')
    required_now: bool = Field(description='当前阶段是否必需')
    active_from: str = Field(description='从哪个阶段开始需要')
    note: str = Field(default='', description='补充说明')


class HealthResponse(BaseModel):
    ok: bool = Field(description='当前阶段所必需的依赖是否全部就绪')
    app_name: str
    app_version: str
    app_env: str
    build_stage: str = Field(description='当前实施阶段，演示时可直接说明进度')
    dependencies: list[DependencyItem]
    # 下面两个字段是给**启动脚本**用的，不是给人看的。
    #
    # 它们回答的是一个很容易被忽略的问题：**"这个服务是什么时候起来的？"**
    #
    # 为什么必须能回答：uvicorn 默认**不会**热加载（没开 --reload），
    # 所以服务一旦起来，就锁定了**那一刻的代码和配置**。
    # 而启动脚本原来只看端口：端口被占用就认为"服务已在运行，跳过启动"。
    # 于是出现过一个真实故障——一个 13 天前启动的旧进程一直占着 8000，
    # 每次"重启"都只是打印了一行"启动完成"，用户访问的还是那个
    # **场景迁移之前**的进程：它的数据库和向量集合都指向迁移前的，
    # 所以两条检索路都返回空，问答一律"无法判断"。
    #
    # 光有 started_at 还不够，脚本要拿它跟**磁盘上代码和配置的最后修改时间**
    # 比一比，才知道这个进程是不是旧的。
    pid: int = Field(description='服务进程的 PID')
    started_at: str = Field(description='服务进程启动时间（ISO 8601）')


class KnobItem(BaseModel):
    """一个可调参数（旋钮）。"""

    group: str = Field(description='所属分类')
    name: str = Field(description='配置项名')
    value: str = Field(description='当前值')
    note: str = Field(default='', description='拧了会影响什么')


class ConfigResponse(BaseModel):
    build_stage: str
    knobs: list[KnobItem]
