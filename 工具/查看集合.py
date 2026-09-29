"""列出向量库里的所有集合及其规模。

为什么需要这个：Zilliz 免费版对**集合数量**有上限（当前是 5 个）。
撞上上限时报的错是 `exceeded the limit number of collections`，
但它不会告诉你"哪个集合是可以删的"。

判断哪个能删，需要三个信息一起看：集合名、向量条数、以及它是不是本项目的。
这个脚本把它们列出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

from app.config import get_settings  # noqa: E402
from app.core.vector_store import get_vector_store  # noqa: E402

# 本项目自己的集合。其它集合是同一台 Zilliz 上别的项目的，不要动。
#
# 判断依据是**配置里当前用的集合名**（下面 main 里从 settings 取），
# 而不是写死一份名单——写死名单的问题是：换了集合名之后，
# 旧的会被标成"别的项目"、新的会被标成"不认识"。
#
# 这里只留**本项目历史用过的名字**，用于识别"哪些是我的、可以清"。
OURS_HISTORY = {'vibe_regulation_knowledge'}


def main() -> int:
    settings = get_settings()
    store = get_vector_store()
    client = store.client

    print(f'向量库：{settings.milvus_uri}')
    print(f'本项目当前使用的集合：{store.collection}')
    print()

    ours = OURS_HISTORY | {store.collection}
    print(f'{"集合名":<26} {"条数":>8}  归属')
    print('-' * 56)
    for name in sorted(client.list_collections()):
        try:
            result = client.query(
                collection_name=name, filter='', output_fields=['count(*)']
            )
            total = int(result[0].get('count(*)', 0)) if result else 0
        except Exception:  # noqa: BLE001
            total = -1
        owner = '← 本项目' if name in ours else '别的项目（不要动）'
        print(f'{name:<26} {total:>8}  {owner}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
