import gc
import statistics
import time

import asnumpy as ap


PROBE_SHAPE = (600_000,)  # ~2.29 MB, large path
BACKLOG_LEVELS = [0, 8, 16, 32]
ROUNDS = 5
WARMUP = 50
ITERS = 500


def reset_runtime():
    ap.refresh_memory_pool_config()
    ap.trim_cache()
    ap.reset_memory_stats()
    gc.collect()


def make_large_backlog(backlog_blocks):
    survivors = []
    temps = []

    # 先分配一串不同大小的 large tensor，尽量制造多个独立 free block
    for i in range(backlog_blocks * 2):
        shape = (600_000 + i * 2048,)  # 每个都略有不同，避免完全同 size
        temps.append(ap.empty(shape, dtype="float32"))

    # 释放一半，保留一半，避免全部 merge 回单个大块
    for i, arr in enumerate(temps):
        if i % 2 == 0:
            survivors.append(arr)
        else:
            del arr

    gc.collect()
    return survivors


def run_probe():
    latencies_us = []
    for _ in range(WARMUP):
        tmp = ap.empty(PROBE_SHAPE, dtype="float32")
        del tmp
    gc.collect()

    ap.reset_memory_stats()
    start = time.perf_counter()
    for _ in range(ITERS):
        t0 = time.perf_counter()
        tmp = ap.empty(PROBE_SHAPE, dtype="float32")
        del tmp
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)
    elapsed = time.perf_counter() - start
    stats = ap.memory_debug_stats("tensor")
    return {
        "median_us": statistics.median(latencies_us),
        "mean_us": statistics.mean(latencies_us),
        "ops_s": ITERS / elapsed,
        "stats": stats,
    }


def main():
    ap.init()
    ap.set_device(0)

    print("=" * 120)
    print("Large Decay Backlog Probe")
    print("=" * 120)
    print(
        f"{'Backlog':<10} {'Median us':>12} {'Mean us':>12} {'Ops/s':>12} "
        f"{'Large hits':>12} {'Large miss':>12} {'Dirty blk':>12} {'Retained blk':>14}"
    )
    print("-" * 120)

    for backlog in BACKLOG_LEVELS:
        medians = []
        means = []
        ops = []
        stats_list = []

        for _ in range(ROUNDS):
            reset_runtime()
            survivors = make_large_backlog(backlog)
            result = run_probe()
            medians.append(result["median_us"])
            means.append(result["mean_us"])
            ops.append(result["ops_s"])
            stats_list.append(result["stats"])
            del survivors
            gc.collect()

        stats = stats_list[len(stats_list) // 2]
        print(
            f"{backlog:<10} "
            f"{statistics.median(medians):>12.2f} "
            f"{statistics.median(means):>12.2f} "
            f"{statistics.median(ops):>12.2f} "
            f"{stats.get('large_cache_hits', 0):>12} "
            f"{stats.get('large_cache_misses', 0):>12} "
            f"{stats.get('dirty_large_block_count', 0):>12} "
            f"{stats.get('retained_large_block_count', 0):>14}"
        )


if __name__ == "__main__":
    main()
