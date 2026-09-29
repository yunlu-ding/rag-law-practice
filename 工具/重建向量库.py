"""重建向量库集合。

什么时候需要它：

Milvus / Zilliz **不支持给已有集合加字段，也不支持改字段类型**。
所以只要向量库的 schema 变了（比如场景迁移时新增了 legal_level、
validity 这几个过滤字段），老集合就不能用了——写入会报 "field not found"。

唯一的路是删掉重建。**这是一次不可逆的数据丢失**，所以：

  - 它被做成了一个独立脚本，而不是藏在启动流程里的"自动修复"；
  - 必须显式加 `--yes` 才真正执行；
  - 执行前会打印集合里现有的条数，让人知道要丢掉多少东西；
  - 重建之后，向量需要重新生成——**embedding 是要花钱的**，
    所以脚本结束时会明确提醒下一步该做什么。

用法：
    python 工具/重建向量库.py              # 只看现状，不动手
    python 工具/重建向量库.py --yes        # 删除并重建
    python 工具/重建向量库.py --yes --drop <另一个集合名>
                                          # 先删掉另一个集合，再建当前集合

`--drop` 存在的原因：Zilliz 免费版对集合**数量**有上限（当前 5 个）。
撞上上限时（`exceeded the limit number of collections`），
要么删掉一个不用的集合，要么升级套餐。
删集合必须**点名**——默认行为只动当前配置的那个集合，
不会因为"顺手清理"把同一台实例上别的项目的数据删掉。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.config import get_settings  # noqa: E402
from app.core.vector_store import get_vector_store  # noqa: E402


def _count(store, name: str) -> int:
    """数一个集合里有多少条向量（删之前先把数字打出来）。"""

    try:
        result = store.client.query(
            collection_name=name, filter='', output_fields=['count(*)']
        )
        return int(result[0].get('count(*)', 0)) if result else 0
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(description='重建向量库集合')
    parser.add_argument('--yes', action='store_true', help='确认删除并重建')
    parser.add_argument(
        '--drop',
        action='append',
        default=[],
        metavar='COLLECTION',
        help='额外删除的集合名（可重复）。用于腾出集合数量配额',
    )
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()
    name = store.collection

    print(f'向量库：{settings.milvus_uri}')
    print(f'集合名：{name}')
    print(f'维度  ：{store.dimension}')
    print()

    try:
        exists = store.client.has_collection(name)
    except Exception as exc:  # noqa: BLE001
        print(f'连不上向量库：{type(exc).__name__}: {exc}')
        return 1

    if exists:
        description = store.client.describe_collection(name)
        fields = [field.get('name') for field in description.get('fields', [])]
        print(f'现有集合字段：{fields}')
        print(f'现有向量条数：{store.count()}')
    else:
        print('集合还不存在。')

    if not args.yes:
        print()
        print('演练模式：什么都没做。确认要重建就加 --yes。')
        return 0

    for name_to_drop in args.drop:
        if not store.client.has_collection(name_to_drop):
            print(f'--drop {name_to_drop}：集合不存在，跳过')
            continue
        print()
        print(f'--drop {name_to_drop}：即将删除，当前条数 {_count(store, name_to_drop)}')
        store.client.drop_collection(name_to_drop)
        print(f'已删除 {name_to_drop}。')

    if exists:
        print()
        print('正在删除集合 ...')
        store.client.drop_collection(name)
        print('已删除。')

    # 删掉之后把进程内的"已确认"标记清掉，否则 ensure_collection 会直接返回，
    # 以为集合还在。
    store._ensured = False  # noqa: SLF001
    store.ensure_collection()

    description = store.client.describe_collection(name)
    fields = [field.get('name') for field in description.get('fields', [])]
    print()
    print(f'已重建。新字段：{fields}')
    print(f'当前条数：{store.count()}')
    print()
    print('下一步：向量是空的，需要重新入库才能生成。')
    print('注意 embedding 调用会产生费用。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
