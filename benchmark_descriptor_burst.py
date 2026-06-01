import gc
import statistics
import time

import asnumpy as ap


WORKLOADS = [
    {
        "name": "empty_small_burst",
        "op": "empty",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
    },
    {
        "name": "empty_large_burst",
        "op": "empty",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
    },
    {
        "name": "zeros_small_burst",
        "op": "zeros",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
    },
    {
        "name": "zeros_large_burst",
        "op": "zeros",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
    },
]


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def bytes_for_shape(shape, itemsize=4):
    total = 1
    for dim in shape:
        total *= dim
    return total * itemsize


def reset_runtime():
    ap.refresh_memory_pool_config()
    ap.trim_cache()
    ap.clear_descriptor_cache()
    ap.reset_memory_stats()
    ap.reset_descriptor_stats()
    gc.collect()


def warm_tensor_pool(size_bytes):
    ap.benchmark_allocator(
        size_bytes=size_bytes,
        iterations=256,
        warmup=64,
        domain="tensor",
    )


def run_workload(workload):
    iterations = workload["iterations"]
    start = time.perf_counter()
    if workload["op"] == "empty":
        for _ in range(iterations):
            temp = ap.empty(workload["shape"], dtype="float32")
            del temp
    elif workload["op"] == "zeros":
        for _ in range(iterations):
            temp = ap.zeros(workload["shape"], dtype="float32")
            del temp
    else:
        raise ValueError(f"Unsupported workload op: {workload['op']}")
    elapsed = time.perf_counter() - start
    return {
        "throughput_ops": iterations / elapsed if elapsed > 0.0 else 0.0,
        "avg_latency_us": (elapsed / iterations) * 1_000_000 if iterations > 0 else 0.0,
        "descriptor_stats": ap.descriptor_stats(),
        "memory_stats": ap.memory_stats("tensor"),
    }


def benchmark_round(workload):
    alloc_bytes = bytes_for_shape(workload["shape"])
    reset_runtime()
    warm_tensor_pool(alloc_bytes)

    ap.clear_descriptor_cache()
    ap.reset_descriptor_stats()
    ap.reset_memory_stats()
    cold = run_workload(workload)

    ap.reset_descriptor_stats()
    ap.reset_memory_stats()
    warm = run_workload(workload)
    return cold, warm


def median_metric(samples, key):
    return statistics.median(sample[key] for sample in samples)


def summarize_samples(samples):
    def summarize_numeric_mapping(items):
        sample = items[0]
        summary = {}
        for key, value in sample.items():
            values = [item[key] for item in items]
            if isinstance(value, dict):
                summary[key] = summarize_numeric_mapping(values)
            else:
                summary[key] = statistics.median(values)
        return summary

    return {
        "throughput_ops": median_metric(samples, "throughput_ops"),
        "avg_latency_us": median_metric(samples, "avg_latency_us"),
        "descriptor_stats": summarize_numeric_mapping([sample["descriptor_stats"] for sample in samples]),
        "memory_stats": summarize_numeric_mapping([sample["memory_stats"] for sample in samples]),
    }


def print_report(workload, cold, warm):
    cold_desc = cold["descriptor_stats"]
    warm_desc = warm["descriptor_stats"]
    speedup = (
        warm["throughput_ops"] / cold["throughput_ops"]
        if cold["throughput_ops"] > 0.0
        else 0.0
    )
    latency_delta_pct = (
        (warm["avg_latency_us"] - cold["avg_latency_us"]) / cold["avg_latency_us"] * 100.0
        if cold["avg_latency_us"] > 0.0
        else 0.0
    )

    print("=" * 104)
    print(
        f"Workload: {workload['name']} | "
        f"op={workload['op']} | "
        f"shape={workload['shape']} | "
        f"alloc={format_bytes(bytes_for_shape(workload['shape']))} | "
        f"rounds={workload['rounds']}"
    )
    print("=" * 104)
    print(
        f"{'Metric':<24} {'Cold Burst':>15} {'Warm Burst':>15} {'Delta':>16}"
    )
    print("-" * 76)
    print(
        f"{'Throughput (ops/s)':<24} "
        f"{cold['throughput_ops']:>15.2f} "
        f"{warm['throughput_ops']:>15.2f} "
        f"{speedup:>12.2f}x"
    )
    print(
        f"{'Latency (us)':<24} "
        f"{cold['avg_latency_us']:>15.2f} "
        f"{warm['avg_latency_us']:>15.2f} "
        f"{latency_delta_pct:>+11.1f}%"
    )
    print("-" * 76)
    print(
        "Descriptor Stats: "
        f"cold hits/misses={cold_desc['hits']}/{cold_desc['misses']}, "
        f"warm hits/misses={warm_desc['hits']}/{warm_desc['misses']}"
    )
    print(
        "Tensor Pool Stats: "
        f"cold hits/misses={cold['memory_stats']['cache_hits']}/{cold['memory_stats']['cache_misses']}, "
        f"warm hits/misses={warm['memory_stats']['cache_hits']}/{warm['memory_stats']['cache_misses']}"
    )
    print()


def print_summary(results):
    print("=" * 132)
    print("Descriptor Burst Summary")
    print("=" * 132)
    print(
        f"{'Workload':<20} {'Cold ops/s':>12} {'Warm ops/s':>12} "
        f"{'Speedup':>10} {'Cold miss':>10} {'Warm hit':>10} {'Warm miss':>10} {'Rounds':>8}"
    )
    print("-" * 132)
    for workload, cold, warm in results:
        cold_desc = cold["descriptor_stats"]
        warm_desc = warm["descriptor_stats"]
        speedup = (
            warm["throughput_ops"] / cold["throughput_ops"]
            if cold["throughput_ops"] > 0.0
            else 0.0
        )
        print(
            f"{workload['name']:<20} "
            f"{cold['throughput_ops']:>12.2f} "
            f"{warm['throughput_ops']:>12.2f} "
            f"{speedup:>9.2f}x "
            f"{cold_desc['misses']:>10} "
            f"{warm_desc['hits']:>10} "
            f"{warm_desc['misses']:>10} "
            f"{workload['rounds']:>8}"
        )
    print()


def main():
    ap.init()
    ap.set_device(0)
    ap.refresh_memory_pool_config()

    results = []
    for workload in WORKLOADS:
        cold_samples = []
        warm_samples = []
        for _ in range(workload["rounds"]):
            cold, warm = benchmark_round(workload)
            cold_samples.append(cold)
            warm_samples.append(warm)
        cold_summary = summarize_samples(cold_samples)
        warm_summary = summarize_samples(warm_samples)
        print_report(workload, cold_summary, warm_summary)
        results.append((workload, cold_summary, warm_summary))

    print_summary(results)


if __name__ == "__main__":
    main()
