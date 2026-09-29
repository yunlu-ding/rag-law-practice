from __future__ import annotations

import logging
import sys

LOG_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def configure_logging(level: int = logging.INFO) -> None:
    """统一配置日志格式。

    为什么显式指定 stdout 并强制 utf-8：
    Windows 下 Python 的标准输出默认使用系统本地编码（中文系统是 GBK），
    日志里一旦出现 emoji 或特殊字符就会抛 UnicodeEncodeError。
    而这个异常发生在"打印日志"这一步，却会中断真正的业务流程——
    一个与业务无关的编码问题，表现起来像是功能坏了。
    这里从根上解决，而不是等出问题再逐个删字符。
    """

    stream = sys.stdout
    try:
        stream.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        # 某些环境（比如被重定向的输出流）不支持 reconfigure，忽略即可
        pass

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 第三方库的日志压到 WARNING，避免刷屏掩盖我们自己的日志。
    #
    # dashscope 单独压到 CRITICAL：它在每次请求失败时都会打印**完整的 Python 调用栈**，
    # 一次失败就是几十行。而我们的代码已经把失败原因整理成一句人话了
    # （比如"检索服务暂时不可用……（原因：向量化连续 3 次失败）"），
    # 那些调用栈除了把有用的日志淹掉之外没有额外信息。
    for noisy in ('httpx', 'httpcore', 'urllib3', 'uvicorn.access'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger('dashscope').setLevel(logging.CRITICAL)
