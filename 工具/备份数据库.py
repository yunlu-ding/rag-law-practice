"""把关系库里的文档与切片导出成 JSON，用于场景迁移前留档。

为什么要有这个脚本：

换语料、重建索引之前会把 document / chunk 两张表清空重灌。
清空是不可逆的——一旦发现新语料有问题想回头对照，旧数据就没了。
导出一次成本极低，所以这类操作前固定做一次。

用法：
    python 工具/备份数据库.py
    python 工具/备份数据库.py --output 工具/_backup_xxx.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

# 让脚本能 import 到 backend/app
BACKEND = Path(__file__).resolve().parent.parent / 'backend'
sys.path.insert(0, str(BACKEND))

from sqlalchemy import text  # noqa: E402

from app.core.postgres import get_engine  # noqa: E402


def _json_default(value):
    """把 psycopg 返回的非 JSON 原生类型转成可序列化的形式。

    日期和 Decimal 都会出现在这两张表里，不处理就会在 json.dump 时炸掉。
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _fetch_rows(table: str) -> list[dict]:
    with get_engine().connect() as conn:
        result = conn.execute(text(f'SELECT * FROM {table}'))  # noqa: S608
        return [dict(row._mapping) for row in result]


def main() -> int:
    parser = argparse.ArgumentParser(description='导出 document / chunk 两张表')
    parser.add_argument(
        '--output',
        default=str(Path(__file__).resolve().parent.parent / 'backend' / 'storage' / '_backup_场景迁移.json'),
        help='输出文件路径',
    )
    args = parser.parse_args()

    document_rows = _fetch_rows('document')
    chunk_rows = _fetch_rows('chunk')

    payload = {
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'source': 'vibe_rag',
        'note': '清库前的留档',
        'counts': {'document': len(document_rows), 'chunk': len(chunk_rows)},
        'document': document_rows,
        'chunk': chunk_rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding='utf-8',
    )

    print(f'已导出 document={len(document_rows)} chunk={len(chunk_rows)}')
    print(f'文件：{output}')
    print(f'大小：{output.stat().st_size / 1024:.1f} KB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
