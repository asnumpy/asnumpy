import gc
import statistics
import time

import asnumpy as ap


WORKLOADS = [
    {
        "name": "zeros_small_executor_burst",
        "op": "zeros",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
        "warmup": 128,
    },
    {
        "name": "zeros_large_executor_burst",
        "op": "zeros",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
        "warmup": 64,
    },
    {
        "name": "ones_small_executor_burst",
        "op": "ones",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
        "warmup": 128,
    },
    {
        "name": "ones_large_executor_burst",
        "op": "ones",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
        "warmup": 64,
    },
    {
        "name": "sum_small_executor_burst",
        "op": "sum",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
        "warmup": 128,
    },
    {
        "name": "sum_large_executor_burst",
        "op": "sum",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
        "warmup": 64,
    },
    {
        "name": "add_small_executor_burst",
        "op": "add",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
        "warmup": 128,
    },
    {
        "name": "add_large_executor_burst",
        "op": "add",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
        "warmup": 64,
    },
    {
        "name": "multiply_small_executor_burst",
        "op": "multiply",
        "shape": (256, 256),
        "iterations": 32,
        "rounds": 7,
        "warmup": 128,
    },
    {
        "name": "multiply_large_executor_burst",
        "op": "multiply",
        "shape": (1024, 1024),
        "iterations": 16,
        "rounds": 7,
        "warmup": 64,
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
    ap.clear_executor_cache()
    ap.reset_memory_stats()
    ap.reset_descriptor_stats()
    ap.reset_executor_stats()
    gc.collect()


def warm_tensor_state(shape, warmup):
    for _ in range(warmup):
        temp = ap.empty(shape, dtype="float32")
        del temp
    gc.collect()


def build_context(workload):
    op = workload["op"]
    if op == "sum":
        return {
            "input": ap.ones(workload["shape"], dtype="float32"),
        }
    if op in ("add", "multiply"):
        return {
            "lhs": ap.ones(workload["shape"], dtype="float32"),
            "rhs": ap.ones(workload["shape"], dtype="float32"),
        }
    return {}


def clear_context(context):
    for key in list(context.keys()):
        del context[key]
    gc.collect()


def run_workload(workload, context):
    op = workload["op"]
    shape = workload["shape"]
    iterations = workload["iterations"]

    start = time.perf_counter()
    for _ in range(iterations):
        if op == "zeros":
            temp = ap.zeros(shape, dtype="float32")
        elif op == "ones":
            temp = ap.ones(shape, dtype="float32")
        elif op == "sum":
            temp = ap.sum(context["input"], axis=0, keepdims=False)
        elif op == "add":
            temp = ap.add(context["lhs"], context["rhs"])
        elif op == "multiply":
            temp = ap.multiply(context["lhs"], context["rhs"])
        else:
            raise ValueError(f"Unsupported op: {op}")
        del temp
    elapsed = time.perf_counter() - start
    return {
        "throughput_ops": iterations / elapsed if elapsed > 0.0 else 0.0,
        "avg_latency_us": (elapsed / iterations) * 1_000_000 if iterations > 0 else 0.0,
        "executor_stats": ap.executor_stats(),
        "descriptor_stats": ap.descriptor_stats(),
        "memory_stats": ap.memory_stats("tensor"),
    }


def benchmark_round(workload):
    reset_runtime()
    warm_tensor_state(workload["shape"], workload["warmup"])
    context = build_context(workload)

    ap.clear_executor_cache()
    ap.reset_executor_stats()
    ap.reset_descriptor_stats()
    ap.reset_memory_stats()
    cold = run_workload(workload, context)

    ap.reset_executor_stats()
    ap.reset_descriptor_stats()
    ap.reset_memory_stats()
    warm = run_workload(workload, context)
    clear_context(context)
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
        "executor_stats": summarize_numeric_mapping([sample["executor_stats"] for sample in samples]),
        "descriptor_stats": summarize_numeric_mapping([sample["descriptor_stats"] for sample in samples]),
        "memory_stats": summarize_numeric_mapping([sample["memory_stats"] for sample in samples]),
    }


def print_report(workload, cold, warm):
    cold_exec = cold["executor_stats"]
    warm_exec = warm["executor_stats"]
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

    print("=" * 110)
    print(
        f"Workload: {workload['name']} | "
        f"shape={workload['shape']} | "
        f"alloc={format_bytes(bytes_for_shape(workload['shape']))} | "
        f"rounds={workload['rounds']}"
    )
    print("=" * 110)
    print(f"{'Metric':<24} {'Cold Burst':>15} {'Warm Burst':>15} {'Delta':>16}")
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
        "Executor Stats: "
        f"cold hits/misses/builds={cold_exec['hits']}/{cold_exec['misses']}/{cold_exec['builds']}, "
        f"warm hits/misses/builds={warm_exec['hits']}/{warm_exec['misses']}/{warm_exec['builds']}"
    )
    print(
        "Descriptor Stats: "
        f"cold hits/misses={cold['descriptor_stats']['hits']}/{cold['descriptor_stats']['misses']}, "
        f"warm hits/misses={warm['descriptor_stats']['hits']}/{warm['descriptor_stats']['misses']}"
    )
    print()


def print_summary(results):
    print("=" * 136)
    print("Executor Burst Summary")
    print("=" * 136)
    print(
        f"{'Workload':<28} {'Cold ops/s':>12} {'Warm ops/s':>12} {'Speedup':>10} "
        f"{'Cold build':>11} {'Warm hit':>10} {'Warm miss':>10} {'Rounds':>8}"
    )
    print("-" * 136)
    for workload, cold, warm in results:
        cold_exec = cold["executor_stats"]
        warm_exec = warm["executor_stats"]
        speedup = (
            warm["throughput_ops"] / cold["throughput_ops"]
            if cold["throughput_ops"] > 0.0
            else 0.0
        )
        print(
            f"{workload['name']:<28} "
            f"{cold['throughput_ops']:>12.2f} "
            f"{warm['throughput_ops']:>12.2f} "
            f"{speedup:>9.2f}x "
            f"{cold_exec['builds']:>11} "
            f"{warm_exec['hits']:>10} "
            f"{warm_exec['misses']:>10} "
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
