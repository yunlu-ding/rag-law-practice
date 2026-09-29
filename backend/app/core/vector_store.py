from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

"""向量库客户端（Milvus / Zilliz Cloud）。

为什么用原生客户端定义显式 schema，而不是用框架的向量库封装：

1. **字段是产品的一部分，不是实现细节。**
   页码、小节标题、切片策略这些字段决定了引用能不能精确到"第几页、哪一条"。
   把它们显式写出来，比藏在框架的元数据字典里更好维护，也更好解释。
2. **删除和过滤要可控。**
   重新索引时要按文档删除旧向量，这依赖 document_id 是一个真实的标量字段。

代价是要自己写 schema 和索引参数，多几十行；
换来的是"这一层发生了什么"完全可见。
"""

# 文本字段的最大长度。Milvus 的 VARCHAR 上限是 65535。
# 超长会被截断，所以入库前要确认切片长度在范围内——
# 我们的目标切片长度是 500 字，硬上限 1000 字，远小于这个值。
MAX_TEXT_LENGTH = 65535
MAX_NAME_LENGTH = 512


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


def serialize_scope(scope: list[str] | None) -> str:
    """把适用范围数组序列化成可 LIKE 匹配的字符串。

    ["证券", "基金"] → ";证券;基金;"
    没有适用范围 → ";"（一个"空的非空值"，这样过滤时语义清楚，
    不会和"字段为空字符串"混在一起）。
    """

    if not scope:
        return ';'
    cleaned = [str(item).strip() for item in scope if str(item).strip()]
    return ';' + ';'.join(cleaned) + ';' if cleaned else ';'


def _connection_args() -> dict[str, Any]:
    settings = get_settings()
    if not settings.milvus_uri:
        raise VectorStoreError(
            '还没有配置向量库。请在 backend/.env 里填 MILVUS_URI；'
            '托管版（Zilliz Cloud）还需要 MILVUS_TOKEN。'
        )
    args: dict[str, Any] = {'uri': settings.milvus_uri}
    if settings.milvus_token:
        args['token'] = settings.milvus_token
    return args


class VectorStore:
    """向量库的读写。

    只暴露四个动作：建集合、写、查、删。
    业务层不需要知道底层是 Milvus 还是别的——
    将来真要换实现，改的是这一个文件。
    """

    def __init__(self) -> None:
        self._client = None
        # 一个进程内只需要把集合结构和索引确认一次。
        # 不加这个标记的话，每个批次写入都会重新检查一遍，
        # 大批量索引时会白白多出几十次远端往返。
        self._ensured = False

    # ---------- 基础设施 ----------

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(**_connection_args())
        return self._client

    @property
    def collection(self) -> str:
        return get_settings().milvus_collection

    @property
    def dimension(self) -> int:
        return get_settings().milvus_dimension

    def _build_schema(self):
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='document_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=MAX_NAME_LENGTH),
            FieldSchema(name='page_number', dtype=DataType.INT64),
            FieldSchema(name='section_title', dtype=DataType.VARCHAR, max_length=MAX_NAME_LENGTH),
            FieldSchema(name='splitter_name', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='content_type', dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            # ---- 场景迁移（法规）新增的过滤字段 ----
            #
            # 为什么这四个字段要进**向量库**，而不是只存在关系库里：
            # 检索过滤发生在向量库里（`filter=...` 是在 Milvus 侧执行的），
            # 如果字段只存在关系库，就得先把向量全捞回来再在应用层筛——
            # 那等于放弃了向量库的过滤能力，Top-K 的 K 会被无关层级的结果稀释掉。
            #
            # 只加这四个、不加全部元数据，也是刻意的：
            # Milvus **不支持给已有集合加字段**，改一次 schema 就要重建整个集合。
            # 所以进向量库的字段必须满足"检索时真的要用来过滤"，
            # 其它字段（文号、发布日期、出处链接）留在关系库，按需 JOIN。
            FieldSchema(name='legal_level', dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name='level_rank', dtype=DataType.INT64),
            FieldSchema(name='validity', dtype=DataType.VARCHAR, max_length=32),
            # scope 存成字符串而不是数组：Milvus 的 ARRAY 过滤语法
            # 在不同版本间不一致，而字符串 LIKE 到处都能用。
            # 序列化格式是 ";证券;基金;"——两端各补一个分隔符，
            # 这样匹配 ";基金;" 既能命中"证券;基金"，又不会误命中
            # 将来可能出现的"基金子公司"这类更长的取值。
            FieldSchema(name='scope', dtype=DataType.VARCHAR, max_length=128),
        ]
        return CollectionSchema(fields, description='金融监管法规知识库切片向量')

    def ensure_collection(self) -> None:
        """集合不存在就创建；已存在则校验维度是否一致。

        维度校验很关键：如果配置里的维度改了而集合还是老的，
        写入会以一个底层错误失败。提前校验能把它变成一句人话，
        顺便提醒"改 embedding 模型要重建整个向量库"这个不可逆的操作。

        创建之后必须**显式加载**：Milvus 里"集合存在"和"集合可用"是两件事，
        没加载的集合会以 `collection not loaded` 报错，
        而这个错误信息看起来像是权限或连接问题，很容易走偏。

        还要**检查索引是否存在**：集合存在但没有索引时，加载会报
        `index not found`。这个状态很容易被搞出来——
        只要有一次建集合的调用在中途失败了，就会留下一个"空壳集合"，
        之后每次加载都失败。所以这里做成自愈的：缺索引就补建，
        而不是要求人工去删集合重来。
        """

        if self._ensured:
            return

        if self.client.has_collection(self.collection):
            description = self.client.describe_collection(self.collection)
            existing_fields = set()
            for field in description.get('fields', []):
                existing_fields.add(field.get('name'))
                if field.get('name') == 'vector':
                    existing_dim = (field.get('params') or {}).get('dim')
                    if existing_dim and int(existing_dim) != int(self.dimension):
                        raise VectorStoreError(
                            f'向量维度不一致：集合里是 {existing_dim}，配置里是 {self.dimension}。'
                            f'换 embedding 模型必须重建整个向量库，不能直接改配置。'
                        )

            # 字段缺失的检查。
            #
            # Milvus **不支持给已有集合加字段**，所以一旦 schema 变了，
            # 老集合就是个"少了几列的表"。不检查的后果是：
            # 写入时报一个底层的 "field not found"，看起来像参数写错了。
            #
            # 这里刻意**只报错、不自愈**。自愈意味着自动删掉整个集合——
            # 那是一次不可逆的数据丢失，不能藏在"确保集合可用"这么温和的名字底下。
            expected = {field.name for field in self._build_schema().fields}
            missing = expected - existing_fields
            if missing:
                raise VectorStoreError(
                    f'集合 {self.collection} 的字段和代码里的 schema 对不上，缺：'
                    f'{"、".join(sorted(missing))}。\n'
                    f'Milvus 不支持给已有集合加字段，只能重建。请运行：\n'
                    f'    python 工具/重建向量库.py --yes\n'
                    f'注意：重建会清空向量库里现有的全部数据。'
                )
            self._ensure_index()
            self._load()
            self._ensured = True
            return

        logger.info('[VECTOR] 创建集合: name=%s dim=%s', self.collection, self.dimension)

        # 索引参数必须用 IndexParams 对象，不能直接传字典。
        # 这是新版 pymilvus 的接口变化——报错信息是
        # "wrong type of argument [index_params]"，看到这句就来这里改。
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='vector',
            index_type='AUTOINDEX',
            metric_type='COSINE',
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=self._build_schema(),
            index_params=index_params,
        )
        self._load()
        self._ensured = True

    def _ensure_index(self) -> None:
        """集合已经有向量字段的索引就跳过，没有就补建。"""

        try:
            existing = self.client.list_indexes(self.collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning('[VECTOR] 查询索引列表失败，尝试补建: %s', exc)
            existing = []

        if existing:
            return

        logger.info('[VECTOR] 集合缺少索引，补建中: name=%s', self.collection)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='vector',
            index_type='AUTOINDEX',
            metric_type='COSINE',
        )
        self.client.create_index(
            collection_name=self.collection,
            index_params=index_params,
        )

    def _load(self) -> None:
        """把集合加载进内存。

        加载本身是幂等的：已经加载过的集合再调一次不会有副作用，
        所以每次 ensure_collection 都调一下，比维护"加载过没有"的状态更省心。
        """

        try:
            self.client.load_collection(self.collection)
        except Exception as exc:  # noqa: BLE001
            # 加载失败不直接抛：调用方紧接着的读写会给出更具体的错误，
            # 而"加载失败"这个信息单独看反而不好定位。
            logger.warning('[VECTOR] 加载集合失败: name=%s error=%s', self.collection, exc)

    # ---------- 写 ----------

    def add(
        self,
        *,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        document_meta: dict[str, Any] | None = None,
    ) -> list[int]:
        """写入切片向量，返回向量库生成的主键。

        主键要回写到关系库（chunk.vector_id），否则将来删除和重建索引时
        就找不到对应的向量，只能整库重建——那正是"孤儿向量"的来源。

        `document_meta` 是文档级的元数据（层级、效力、适用范围）。
        它不放在每个 chunk 里传，是因为这些字段**整份文档都一样**——
        让调用方逐片重复传一遍，只是给"某一片的层级写错了"留机会。
        """

        if not chunks:
            return []
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f'切片数与向量数不一致：{len(chunks)} vs {len(vectors)}'
            )

        self.ensure_collection()

        meta = document_meta or {}
        legal_level = str(meta.get('legal_level') or 'unknown')[:32]
        level_rank = int(meta.get('level_rank') or 0)
        validity = str(meta.get('validity') or 'effective')[:32]
        scope = serialize_scope(meta.get('scope'))[:128]

        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows.append(
                {
                    'vector': vector,
                    'chunk_id': str(chunk.get('chunk_id') or ''),
                    'document_id': str(chunk.get('document_id') or ''),
                    'filename': str(chunk.get('filename') or '')[:MAX_NAME_LENGTH],
                    'page_number': int(chunk.get('page_number') or 0),
                    'section_title': str(chunk.get('section_title') or '')[:MAX_NAME_LENGTH],
                    'splitter_name': str(chunk.get('splitter_name') or '')[:64],
                    'content_type': str(chunk.get('content_type') or 'text')[:32],
                    'text': str(chunk.get('text') or '')[:MAX_TEXT_LENGTH],
                    'legal_level': legal_level,
                    'level_rank': level_rank,
                    'validity': validity,
                    'scope': scope,
                }
            )

        result = self.client.insert(collection_name=self.collection, data=rows)
        ids = list(result.get('ids') or [])
        logger.info('[VECTOR] 写入完成: collection=%s count=%s', self.collection, len(ids))
        return [int(item) for item in ids]

    # ---------- 删 ----------

    def delete_by_document(self, document_id: str) -> int:
        """按文档删除向量，返回删除条数。

        文档被重新切分或删除时都要调用它，否则旧向量会留在库里，
        表现为"搜得到、但点开发现内容已经不存在了"——
        这类孤儿数据在检索里极难排查。

        ⚠️ 这里刻意**不像其他降级路径那样"记个日志就继续"**。

        原因是实测踩过一次：删除失败的异常被吞掉、流程继续往下走，
        于是同一份文档在库里留下了两套向量——
        一次运行之后集合里有 1911 条向量，而实际只需要 1047 条，
        多出来的 864 条全是孤儿。

        更糟的是它**不报错**：检索时会返回重复内容，
        用户只会觉得"怎么同一条出现了两次"，根本查不到原因。

        所以规则是：**会影响数据正确性的失败，必须让流程停下来。**
        "降级优于中断"只适用于体验层（比如重排失败退回原顺序），
        不适用于数据层。
        """

        if not self.client.has_collection(self.collection):
            return 0

        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                result = self.client.delete(
                    collection_name=self.collection,
                    filter=f'document_id == "{document_id}"',
                )
                deleted = int((result or {}).get('delete_count', 0))
                logger.info(
                    '[VECTOR] 已删除文档向量: document_id=%s 条数=%s',
                    document_id,
                    deleted,
                )
                return deleted
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    '[VECTOR] 删除文档向量失败（第 %s 次）: document_id=%s error=%s',
                    attempt,
                    document_id,
                    exc,
                )

        raise VectorStoreError(
            f'删除文档向量失败，已中止索引以免产生重复数据：'
            f'document_id={document_id} error={last_error}'
        )

    # ---------- 查 ----------

    def search(
        self,
        *,
        vector: list[float],
        top_k: int = 5,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量相似度检索。

        `filter_expr` 是 Milvus 的标量过滤表达式，例如
        `validity == "effective"` 或 `level_rank <= 3`。
        过滤在**向量库侧**执行，不是在应用层筛结果——
        这是它和"查完再筛"的本质区别：后者会让无关层级的切片
        先占掉 Top-K 的名额，真正想要的反而排不进来。
        """

        if not self.client.has_collection(self.collection):
            return []

        fields = output_fields or [
            'chunk_id', 'document_id', 'filename', 'page_number',
            'section_title', 'splitter_name', 'content_type', 'text',
            'legal_level', 'level_rank', 'validity', 'scope',
        ]
        arguments: dict[str, Any] = {
            'collection_name': self.collection,
            'data': [vector],
            'limit': top_k,
            'output_fields': fields,
            'search_params': {'metric_type': 'COSINE', 'params': {}},
        }
        if filter_expr:
            arguments['filter'] = filter_expr

        results = self.client.search(**arguments)
        if not results:
            return []

        hits: list[dict[str, Any]] = []
        for rank, item in enumerate(results[0], start=1):
            entity = dict(item.get('entity') or {})
            entity['vector_id'] = item.get('id')
            entity['score'] = float(item.get('distance') or 0.0)
            entity['rank_vector'] = rank
            entity['retrieval_source'] = 'vector'
            hits.append(entity)
        return hits

    def count(self) -> int:
        """集合里的向量条数。用于自检与对账。"""

        if not self.client.has_collection(self.collection):
            return 0
        try:
            result = self.client.query(
                collection_name=self.collection,
                filter='',
                output_fields=['count(*)'],
            )
            if result:
                return int(result[0].get('count(*)', 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning('[VECTOR] 统计条数失败: %s', exc)
        return 0


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """返回向量库单例。"""

    return VectorStore()


def vector_store_health() -> tuple[bool, str | None]:
    """探活。返回 (是否可用, 错误信息)。"""

    settings = get_settings()
    if not settings.milvus_uri:
        return False, '未配置 MILVUS_URI'
    try:
        store = get_vector_store()
        store.client.list_collections()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
