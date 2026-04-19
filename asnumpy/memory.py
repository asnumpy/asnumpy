from loguru import logger

from .lib import asnumpy_core as _core


@logger.catch
def clear_cache(domain=None):
    return _core.memory.clear_cache(domain)


@logger.catch
def trim_cache(domain=None):
    return _core.memory.trim(domain)


@logger.catch
def refresh_memory_pool_config():
    return _core.memory.refresh_config()


@logger.catch
def reset_memory_stats(domain=None):
    return _core.memory.reset_stats(domain)


@logger.catch
def memory_stats(domain=None, verbose=False):
    return dict(_core.memory.stats(domain, verbose))


@logger.catch
def memory_debug_stats(domain=None):
    return dict(_core.memory.debug_stats(domain))


@logger.catch
def benchmark_allocator(size_bytes, iterations, warmup=100, domain=None):
    return dict(_core.memory.benchmark_allocator(size_bytes, iterations, warmup, domain))


@logger.catch
def descriptor_stats():
    return dict(_core.memory.descriptor_stats())


@logger.catch
def clear_descriptor_cache():
    return _core.memory.clear_descriptor_cache()


@logger.catch
def reset_descriptor_stats():
    return _core.memory.reset_descriptor_stats()


@logger.catch
def executor_stats():
    return dict(_core.memory.executor_stats())


@logger.catch
def clear_executor_cache():
    return _core.memory.clear_executor_cache()


@logger.catch
def reset_executor_stats():
    return _core.memory.reset_executor_stats()
