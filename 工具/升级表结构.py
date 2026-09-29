"""给已存在的表补字段。

为什么需要这个工具：

SQLAlchemy 的 `create_all` **只创建不存在的表，不会给已存在的表加字段**。
所以改了模型之后，如果忘了执行 ALTER TABLE，代码里明明有 `legal_level`，
数据库里却没有——报错信息是 `column document.legal_level does not exist`，
看起来像代码写错了，实际是库没跟上。

这个坑在搭建过程中踩过两次（一次是接数据库时，一次是场景迁移时）。
与其每次手写一段 ALTER TABLE 再删掉，不如做成一个可以反复运行的工具：

    读模型 → 查库里的实际字段 → 差集就是要加的列 → 逐条 ALTER

它是**幂等**的：已经存在的列会跳过，重复运行不会有副作用。

⚠️ 它的能力边界要说清楚：**只加列，不改类型、不删列、不动数据。**
改类型和删列是不可逆操作，需要单独的、有备份的迁移脚本，不能顺手做。

用法：
    python 工具/升级表结构.py            # 只看要改什么（默认演练）
    python 工具/升级表结构.py --apply    # 真正执行
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / 'backend'
sys.path.insert(0, str(BACKEND))

# Windows 控制台默认按 GBK 解码，而这个脚本会打印 ✅/⚠️ 这类符号——
# 不加保护的话，它会**直接崩在打印那一步**，而崩溃点常常在干完活之后
# （评测跑完了、钱花完了，明细一条都没落盘）。详见 工具/修控制台编码.py。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from sqlalchemy import inspect, text  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402

from app.core.postgres import get_engine  # noqa: E402
from app.models import Base  # noqa: E402

DIALECT = postgresql.dialect()


def column_statements(table_name: str, column) -> list[str]:
    """把模型里的一个字段翻译成一组 ALTER TABLE 语句。"""

    type_sql = column.type.compile(dialect=DIALECT)
    statement = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {type_sql}'

    # 有默认值就带上。否则已有的行会是 NULL，而模型里这些字段声明的是
    # "非空 + 有默认"，两边就对不上了——后面写数据时才会有诡异的行为。
    literal: str | None = None
    if column.server_default is not None:
        literal = str(column.server_default.arg)
    elif column.default is not None and not callable(column.default.arg):
        value = column.default.arg
        literal = f"'{value}'" if isinstance(value, str) else str(value)

    statements = [statement]

    if not column.nullable:
        # 加非空列必须给已有行一个值。没有默认值时先允许为空，
        # 由业务补齐数据之后再收紧——比在迁移脚本里瞎填一个值安全。
        if literal is not None:
            statements[0] += f' DEFAULT {literal}'
            statements.append(
                f'ALTER TABLE "{table_name}" ALTER COLUMN "{column.name}" SET NOT NULL'
            )
        else:
            print(f'    ⚠️ {column.name} 声明为非空但没有默认值，先按可空添加')

    if column.comment:
        escaped = column.comment.replace("'", "''")
        statements.append(
            f'COMMENT ON COLUMN "{table_name}"."{column.name}" IS \'{escaped}\''
        )

    return statements


def main() -> int:
    parser = argparse.ArgumentParser(description='给已存在的表补字段（幂等）')
    parser.add_argument('--apply', action='store_true', help='真正执行；不加则只演练')
    args = parser.parse_args()

    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    planned: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            print(f'[跳过] 表 {table.name} 还不存在，create_all 会建它')
            continue

        actual = {column['name'] for column in inspector.get_columns(table.name)}
        missing = [column for column in table.columns if column.name not in actual]
        if not missing:
            print(f'[一致] {table.name}：{len(actual)} 个字段，无需变更')
            continue

        print(f'[待补] {table.name}：缺 {len(missing)} 个字段')
        for column in missing:
            statements = column_statements(table.name, column)
            # 打印**编译后**的类型，而不是类型对象的 repr。
            # 两者不一样：DateTime(timezone=True) 的 repr 是 "DATETIME"，
            # 但真正发给 PostgreSQL 的是 "TIMESTAMP WITH TIME ZONE"。
            # 照着 repr 去核对，会以为这一步在造一个 PG 不存在的类型。
            shown = column.type.compile(dialect=DIALECT)
            print(f'    + {column.name}  ({shown})')
            planned.extend(statements)

    if not planned:
        print()
        print('没有需要补的字段。')
        return 0

    if not args.apply:
        print()
        print(f'演练结束，共 {len(planned)} 条 ALTER 待执行。加 --apply 真正执行。')
        return 0

    print()
    print('执行中 ...')
    with engine.begin() as connection:
        for statement in planned:
            connection.execute(text(statement))
    print(f'完成，共执行 {len(planned)} 条 ALTER 语句。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
