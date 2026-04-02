from loguru import logger

from .lib import asnumpy_core as _core


@logger.catch
def clear_cache():
    return _core.memory.clear_cache()


@logger.catch
def trim_cache():
    return _core.memory.trim()


@logger.catch
def memory_stats():
    return dict(_core.memory.stats())
