from __future__ import annotations

from app.rag.splitters.base import (
    DEFAULT_CHUNK_SIZE,
    SENTENCE_SEPARATORS,
    SplitChunk,
    find_best_split_point,
)


def split_unstructured_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 50,
) -> list[SplitChunk]:
    """按长度切分，并保留重叠。

    这是最朴素、也是最常见的做法，把它保留下来有两个原因：

    1. **它是兜底方案**。文档完全没有结构标记时，总得有个办法切。
    2. **它是对照组**。只有把它和结构感知切分放在一起跑，
       才能用数字说明"结构感知到底好在哪"，
       而不是凭感觉说"语义切分更高级"。

    它的固有缺陷也很清楚：**重叠是为缺陷打的补丁**。
    因为会切断句子和单词，所以要用 50 字的重复内容把上下文补回来。
    结构感知切分切出来的每片本身就是完整语义单元，就不需要这个补丁。
    """

    if not text:
        return []

    pieces: list[SplitChunk] = []
    position = 0
    total = len(text)

    while position < total:
        end = min(position + chunk_size, total)
        if end < total:
            window = text[position:end]
            best = -1
            for separator in SENTENCE_SEPARATORS:
                index = window.rfind(separator, int(chunk_size * 0.5))
                if index >= 0:
                    best = max(best, index + len(separator))
            if best > 0:
                end = position + best

        pieces.append(SplitChunk(content=text[position:end], start_offset=position, end_offset=end))
        if end >= total:
            break
        # 用重叠把被切断的上下文补回来，同时保证位置一定前进
        position = max(end - overlap, position + 1)

    return pieces


__all__ = ['split_unstructured_text', 'find_best_split_point']
