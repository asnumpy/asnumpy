import gc
import os
import statistics
import time

import asnumpy as ap

ap.init()

WORKLOADS = [
    {
        "name": "zeros_small_path",
        "shape": (256, 256),
        "op": "zeros",
        "iterations": 5000,
        "warmup": 500,
        "rounds": 5,
    },
    {
        "name": "zeros_large_path",
        "shape": (1024, 1024),
        "op": "zeros",
        "iterations": 3000,
        "warmup": 300,
        "rounds": 5,
    },
    {
        "name": "sum_small_path",
        "shape": (256, 256),
        "op": "sum",
        "iterations": 5000,
        "warmup": 500,
        "rounds": 5,
    },
    {
        "name": "sum_large_path",
        "shape": (1024, 1024),
        "op": "sum",
        "iterations": 3000,
        "warmup": 300,
        "rounds": 5,
    },
    {
        "name": "add_small_path",
        "shape": (256, 256),
        "op": "add",
        "iterations": 5000,
        "warmup": 500,
        "rounds": 5,
    },
    {
        "name": "add_large_path",
        "shape": (1024, 1024),
        "op": "add",
        "iterations": 3000,
        "warmup": 300,
        "rounds": 5,
    },
    {
        "name": "multiply_small_path",
        "shape": (256, 256),
        "op": "multiply",
        "iterations": 5000,
        "warmup": 500,
        "rounds": 5,
    },
    {
        "name": "multiply_large_path",
        "shape": (1024, 1024),
        "op": "multiply",
        "iterations": 3000,
        "warmup": 300,
        "rounds": 5,
    },
    {
        "name": "mixed_small_path",
        "shape": (256, 256),
        "op": "mixed",
        "iterations": 5000,
        "warmup": 500,
        "rounds": 5,
    },
    {
        "name": "mixed_large_path",
        "shape": (1024, 1024),
        "op": "mixed",
        "iterations": 3000,
        "warmup": 300,
        "rounds": 5,
    },
]

MODES = [
    {
        "name": "raw_runtime",
        "label": "Raw Runtime",
        "pool_enabled": False,
        "descriptor_reuse": False,
        "executor_reuse": False,
    },
    {
        "name": "pool_only",
        "label": "Pool Only",
        "pool_enabled": True,
        "descriptor_reuse": False,
        "executor_reuse": False,
    },
    {
        "name": "pool_descriptor",
        "label": "Pool+Desc",
        "pool_enabled": True,
        "descriptor_reuse": True,
        "executor_reuse": False,
    },
    {
        "name": "full_stack",
        "label": "Full Stack",
        "pool_enabled": True,
        "descriptor_reuse": True,
        "executor_reuse": True,
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


def reset_runtime_state(mode):
    os.environ["ASN_ENABLE_POOL"] = "1" if mode["pool_enabled"] else "0"
    os.environ["ASN_DEBUG_LOG"] = "0"
    if hasattr(ap, "refresh_memory_pool_config"):
        ap.refresh_memory_pool_config()
    if hasattr(ap, "trim_cache"):
        ap.trim_cache()
    if hasattr(ap, "clear_descriptor_cache"):
        ap.clear_descriptor_cache()
    if hasattr(ap, "clear_executor_cache"):
        ap.clear_executor_cache()
    if hasattr(ap, "reset_memory_stats"):
        ap.reset_memory_stats()
    if hasattr(ap, "reset_descriptor_stats"):
        ap.reset_descriptor_stats()
    if hasattr(ap, "reset_executor_stats"):
        ap.reset_executor_stats()
    gc.collect()


def reset_mode_counters():
    if hasattr(ap, "reset_memory_stats"):
        ap.reset_memory_stats()
    if hasattr(ap, "reset_descriptor_stats"):
        ap.reset_descriptor_stats()
    if hasattr(ap, "reset_executor_stats"):
        ap.reset_executor_stats()


def maybe_disable_reuse(mode):
    if not mode["descriptor_reuse"] and hasattr(ap, "clear_descriptor_cache"):
        ap.clear_descriptor_cache()
    if not mode["executor_reuse"] and hasattr(ap, "clear_executor_cache"):
        ap.clear_executor_cache()


def collect_runtime_stats():
    result = {}
    if hasattr(ap, "memory_stats"):
        result["tensor"] = ap.memory_stats("tensor")
        result["workspace"] = ap.memory_stats("workspace")
    if hasattr(ap, "descriptor_stats"):
        result["descriptor"] = ap.descriptor_stats()
    if hasattr(ap, "executor_stats"):
        result["executor"] = ap.executor_stats()
    return result


def prepare_workload_context(workload):
    if workload["op"] in {"sum", "mixed"}:
        return {"input": ap.ones(workload["shape"], dtype="float32")}
    if workload["op"] in {"add", "multiply"}:
        return {
            "lhs": ap.ones(workload["shape"], dtype="float32"),
            "rhs": ap.ones(workload["shape"], dtype="float32"),
        }
    return {}


def release_workload_context(context):
    context.clear()
    gc.collect()


def run_workload(workload, iterations, context, mode):
    start = time.perf_counter()
    operations = iterations

    if workload["op"] == "zeros":
        for _ in range(iterations):
            maybe_disable_reuse(mode)
            temp = ap.zeros(workload["shape"], dtype="float32")
            del temp
    elif workload["op"] == "sum":
        input_array = context["input"]
        for _ in range(iterations):
            maybe_disable_reuse(mode)
            temp = ap.sum(input_array, axis=0, keepdims=False)
            del temp
    elif workload["op"] == "add":
        lhs = context["lhs"]
        rhs = context["rhs"]
        for _ in range(iterations):
            maybe_disable_reuse(mode)
            temp = ap.add(lhs, rhs)
            del temp
    elif workload["op"] == "multiply":
        lhs = context["lhs"]
        rhs = context["rhs"]
        for _ in range(iterations):
            maybe_disable_reuse(mode)
            temp = ap.multiply(lhs, rhs)
            del temp
    elif workload["op"] == "mixed":
        input_array = context["input"]
        operations = iterations * 2
        for _ in range(iterations):
            maybe_disable_reuse(mode)
            temp1 = ap.zeros(workload["shape"], dtype="float32")
            maybe_disable_reuse(mode)
            temp2 = ap.sum(input_array, axis=0, keepdims=False)
            del temp1
            del temp2
    else:
        raise ValueError(f"Unsupported workload op: {workload['op']}")

    elapsed = time.perf_counter() - start
    throughput = operations / elapsed if elapsed > 0.0 else 0.0
    avg_latency_us = (elapsed / operations) * 1_000_000 if operations > 0 else 0.0
    return {
        "elapsed_s": elapsed,
        "operations": operations,
        "throughput_ops": throughput,
        "avg_latency_us": avg_latency_us,
    }


def benchmark_mode_round(mode, workload):
    reset_runtime_state(mode)

    context = prepare_workload_context(workload)
    run_workload(workload, workload["warmup"], context, mode)
    release_workload_context(context)

    reset_runtime_state(mode)
    context = prepare_workload_context(workload)
    reset_mode_counters()
    result = run_workload(workload, workload["iterations"], context, mode)
    result["stats"] = collect_runtime_stats()
    release_workload_context(context)
    reset_runtime_state(mode)
    return result


def summarize_rounds(rounds):
    first = rounds[0]

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
        "throughput_ops": statistics.median(item["throughput_ops"] for item in rounds),
        "avg_latency_us": statistics.median(item["avg_latency_us"] for item in rounds),
        "elapsed_s": statistics.median(item["elapsed_s"] for item in rounds),
        "operations": first["operations"],
        "stats": summarize_numeric_mapping([item["stats"] for item in rounds]),
    }


def benchmark_mode(mode, workload):
    rounds = []
    for round_idx in range(workload["rounds"]):
        print(f"  [{workload['name']}] {mode['name']} round {round_idx + 1}/{workload['rounds']}...")
        rounds.append(benchmark_mode_round(mode, workload))
    summary = summarize_rounds(rounds)
    summary["mode"] = mode
    return summary


def print_report(workload, results):
    baseline = results["raw_runtime"]

    print("=" * 132)
    print(
        f"Workload: {workload['name']} | "
        f"op={workload['op']} | "
        f"shape={workload['shape']} | "
        f"alloc={format_bytes(bytes_for_shape(workload['shape']))} | "
        f"rounds={workload['rounds']}"
    )
    print("=" * 132)
    print(
        f"{'Mode':<16} {'Ops/s':>12} {'Avg us':>12} {'Speedup':>10} "
        f"{'Tensor HM':>14} {'Desc HM':>14} {'Exec HMB':>18}"
    )
    print("-" * 132)

    for mode in MODES:
        result = results[mode["name"]]
        stats = result.get("stats", {})
        tensor = stats.get("tensor", {})
        descriptor = stats.get("descriptor", {})
        executor = stats.get("executor", {})
        speedup = (
            result["throughput_ops"] / baseline["throughput_ops"]
            if baseline["throughput_ops"] > 0.0
            else 0.0
        )
        print(
            f"{mode['label']:<16} "
            f"{result['throughput_ops']:>12.2f} "
            f"{result['avg_latency_us']:>12.2f} "
            f"{speedup:>9.2f}x "
            f"{tensor.get('cache_hits', 0)}/{tensor.get('cache_misses', 0):<7} "
            f"{descriptor.get('hits', 0)}/{descriptor.get('misses', 0):<7} "
            f"{executor.get('hits', 0)}/{executor.get('misses', 0)}/{executor.get('builds', 0):<7}"
        )
    print()


def print_summary(items):
    print("=" * 152)
    print("Layered End-to-End Summary")
    print("=" * 152)
    print(
        f"{'Workload':<18} {'Raw us':>10} {'Pool us':>10} {'Pool+Desc us':>13} {'Full us':>10} "
        f"{'Pool Gain':>11} {'Desc Gain':>11} {'Exec Gain':>11} {'Full Gain':>11}"
    )
    print("-" * 152)

    for item in items:
        raw = item["results"]["raw_runtime"]["avg_latency_us"]
        pool_only = item["results"]["pool_only"]["avg_latency_us"]
        pool_desc = item["results"]["pool_descriptor"]["avg_latency_us"]
        full = item["results"]["full_stack"]["avg_latency_us"]

        pool_gain = ((pool_only - raw) / raw * 100.0) if raw > 0 else 0.0
        desc_gain = ((pool_desc - pool_only) / pool_only * 100.0) if pool_only > 0 else 0.0
        exec_gain = ((full - pool_desc) / pool_desc * 100.0) if pool_desc > 0 else 0.0
        full_gain = ((full - raw) / raw * 100.0) if raw > 0 else 0.0

        print(
            f"{item['name']:<18} "
            f"{raw:>10.2f} "
            f"{pool_only:>10.2f} "
            f"{pool_desc:>13.2f} "
            f"{full:>10.2f} "
            f"{pool_gain:>+10.1f}% "
            f"{desc_gain:>+10.1f}% "
            f"{exec_gain:>+10.1f}% "
            f"{full_gain:>+10.1f}%"
        )
    print()


def benchmark_workload(workload):
    mode_results = {}
    for mode in MODES:
        mode_results[mode["name"]] = benchmark_mode(mode, workload)
    print_report(workload, mode_results)
    return {
        "name": workload["name"],
        "shape": workload["shape"],
        "alloc_bytes": bytes_for_shape(workload["shape"]),
        "results": mode_results,
    }


def main():
    all_results = []
    for workload in WORKLOADS:
        all_results.append(benchmark_workload(workload))
    print_summary(all_results)


if __name__ == "__main__":
    main()
