import gc
import os
import time

import asnumpy as ap

ap.init()

WORKLOADS = [
    {
        "name": "small_path",
        "shape": (256, 256),  # 256 KB float32, should stay on the small-allocation path
        "iterations": 5000,
        "warmup": 500,
    },
    {
        "name": "large_path",
        "shape": (1024, 1024),  # 4 MB float32, should stay on the large-allocation path
        "iterations": 3000,
        "warmup": 300,
    },
]


def bytes_for_shape(shape, itemsize=4):
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * itemsize


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def reset_pool_state():
    if hasattr(ap, "trim_cache"):
        ap.trim_cache()
    gc.collect()


def run_workload(shape, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        temp = ap.zeros(shape, dtype="float32")
        del temp
    elapsed = time.perf_counter() - start
    throughput = iterations / elapsed
    avg_latency_us = (elapsed / iterations) * 1_000_000
    return {
        "elapsed_s": elapsed,
        "throughput_ops": throughput,
        "avg_latency_us": avg_latency_us,
    }


def benchmark_mode(label, pool_enabled, shape, iterations, warmup):
    os.environ["ASN_ENABLE_POOL"] = "1" if pool_enabled else "0"
    os.environ["ASN_DEBUG_LOG"] = "0"
    reset_pool_state()

    print(f"--- Testing {label} ---")
    print(f"[{label}] Warming up...")
    run_workload(shape, warmup)
    reset_pool_state()

    print(f"[{label}] Running {iterations} iterations...")
    result = run_workload(shape, iterations)
    print(
        f"Result: {result['throughput_ops']:.2f} ops/s | "
        f"Latency: {result['avg_latency_us']:.2f} us"
    )
    print()
    reset_pool_state()
    return result


def print_report(workload, baseline, pooled):
    throughput_speedup = pooled["throughput_ops"] / baseline["throughput_ops"]
    latency_delta_pct = (
        (pooled["avg_latency_us"] - baseline["avg_latency_us"])
        / baseline["avg_latency_us"]
        * 100
    )
    latency_delta_us = pooled["avg_latency_us"] - baseline["avg_latency_us"]

    print("=" * 60)
    print(f"Workload: {workload['name']}")
    print(
        f"Shape: {workload['shape']} | "
        f"Allocation Size: {format_bytes(bytes_for_shape(workload['shape']))}"
    )
    print("=" * 60)
    print()
    print("--- Final Report ---")
    print(
        f"{'Metric':<20} | {'System Call':>12} | {'Memory Pool':>12} | {'Improvement':>16}"
    )
    print("-" * 70)
    print(
        f"{'Throughput (ops/s)':<20} | "
        f"{baseline['throughput_ops']:>12.2f} | "
        f"{pooled['throughput_ops']:>12.2f} | "
        f"{throughput_speedup:>10.2f}x"
    )
    print(
        f"{'Latency (us)':<20} | "
        f"{baseline['avg_latency_us']:>12.2f} | "
        f"{pooled['avg_latency_us']:>12.2f} | "
        f"{latency_delta_pct:>8.1f}% ({latency_delta_us:+.2f} us)"
    )
    print()


def benchmark_workload(workload):
    print()
    print("#" * 72)
    print(
        f"Benchmarking {workload['name']} | "
        f"shape={workload['shape']} | "
        f"alloc={format_bytes(bytes_for_shape(workload['shape']))}"
    )
    print("#" * 72)
    print()

    baseline = benchmark_mode(
        label="System Call (Baseline)",
        pool_enabled=False,
        shape=workload["shape"],
        iterations=workload["iterations"],
        warmup=workload["warmup"],
    )
    pooled = benchmark_mode(
        label="Memory Pool (Optimized)",
        pool_enabled=True,
        shape=workload["shape"],
        iterations=workload["iterations"],
        warmup=workload["warmup"],
    )
    print_report(workload, baseline, pooled)


def main():
    for workload in WORKLOADS:
        benchmark_workload(workload)


if __name__ == "__main__":
    main()
