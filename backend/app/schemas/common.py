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


class KnobItem(BaseModel):
    """一个可调参数（旋钮）。"""

    group: str = Field(description='所属分类')
    name: str = Field(description='配置项名')
    value: str = Field(description='当前值')
    note: str = Field(default='', description='拧了会影响什么')


class ConfigResponse(BaseModel):
    build_stage: str
    knobs: list[KnobItem]
