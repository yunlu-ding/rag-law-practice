from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

"""人工核对的元数据覆盖层。

元数据是正则从正文里猜出来的，**必然有猜错的时候**。改数据库能让当下正确，
但下一次重新入库又会变回错的——而"重新入库"在这个项目里是常规操作
（换语料、改切分策略、重建向量库都会触发）。

所以人工结论要固化成文件，由入库流程去读。这样"人工核对"是一次性的投入，
不是每次重做一遍的负担。

覆盖文件长这样（vibe-rag/语料元数据核对.json）：

    {
      "核对人": "…", "核对基准日": "…",
      "文档": {
        "某法规.pdf": {"effective_date": "2020-03-01", "note": "…"}
      }
    }

两个刻意的设计：

1. **按文件名匹配，不按内容哈希。** 内容哈希会随解析器的每次改动而变化
   （清洗规则改一下，哈希就变了），用它当键的话，人工核对会频繁失效。
   文件名是稳定的。

2. **显式的 null 表示"人工确认过：这个值应当为空"**，与"文件里没写这个字段"
   区分开。前者要覆盖掉自动抽到的错误值，后者要保持不动。

   典型场景：《期货交易管理条例》自动抽到的施行日是 1999-06-02，
   那是它取代的旧暂行条例的日期（正文回述历史时提到的）。
   人工确认"本版施行日待定"之后，就该把那个错值清掉——
   **留一个空值远比留一个看起来像样的错值安全**，
   因为空值会出现在"待人工复核"清单里，错值不会。
"""


def _resolve_path() -> Path:
    # backend/app/rag/ → 上溯三级到 vibe-rag/
    return Path(__file__).resolve().parents[3] / '语料元数据核对.json'


@lru_cache(maxsize=1)
def load_overrides() -> dict[str, Any]:
    """读取覆盖文件。文件不存在时返回空表（不报错）。"""

    path = _resolve_path()
    if not path.exists():
        logger.info('[OVERRIDE] 没有元数据核对文件，跳过：%s', path)
        return {'文档': {}}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        # 覆盖文件本身坏掉时不能静默忽略——那会让所有人工核对悄悄失效。
        logger.error('[OVERRIDE] 元数据核对文件解析失败：%s', exc)
        raise


def apply_overrides(metadata: dict[str, Any], filename: str) -> dict[str, Any]:
    """把人工核对的结论盖到自动抽取的结果上。"""

    table = load_overrides().get('文档', {})
    override = table.get(filename)
    if not override:
        return metadata

    who = load_overrides().get('核对人', '人工')
    when = load_overrides().get('核对基准日', '')
    note = override.get('note') or ''
    stamp = f'人工核对（{who}，{when}）'

    evidence = dict(metadata.get('evidence') or {})
    applied: list[str] = []

    for field, value in override.items():
        if field == 'note':
            continue
        metadata[field] = value
        applied.append(field)
        evidence[field] = f'{stamp}：{value if value is not None else "（确认应为空）"}。{note}'

    # 层级被人工改了，位阶要跟着重算——否则会出现"层级是法律、位阶是自律规则"这种矛盾。
    if 'legal_level' in applied:
        from app.rag.metadata import level_rank

        metadata['level_rank'] = level_rank(metadata.get('legal_level'))

    metadata['evidence'] = evidence
    logger.info('[OVERRIDE] 已应用人工核对: file=%s 字段=%s', filename, applied)
    return metadata


__all__ = ['apply_overrides', 'load_overrides']
