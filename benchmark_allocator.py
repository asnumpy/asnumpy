import asnumpy as ap

ap.init()
ap.set_device(0)
ap.refresh_memory_pool_config()


WORKLOADS = [
    {
        "name": "tensor_small",
        "size_bytes": 256 * 1024,
        "domain": "tensor",
        "iterations": 20000,
        "warmup": 1000,
    },
    {
        "name": "tensor_large",
        "size_bytes": 4 * 1024 * 1024,
        "domain": "tensor",
        "iterations": 10000,
        "warmup": 500,
    },
    {
        "name": "workspace_small",
        "size_bytes": 128 * 1024,
        "domain": "workspace",
        "iterations": 20000,
        "warmup": 1000,
    },
    {
        "name": "workspace_large",
        "size_bytes": 4 * 1024 * 1024,
        "domain": "workspace",
        "iterations": 10000,
        "warmup": 500,
    },
]


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def print_report(item, result):
    pool_stats = result["pool_stats"]
    print("=" * 88)
    print(
        f"Workload: {item['name']} | "
        f"domain={item['domain']} | "
        f"alloc={format_bytes(item['size_bytes'])}"
    )
    print("=" * 88)
    print(
        f"{'Metric':<22} {'System Call':>15} {'Memory Pool':>15} {'Improvement':>18}"
    )
    print("-" * 74)
    print(
        f"{'Throughput (ops/s)':<22} "
        f"{result['baseline_ops_per_sec']:>15.2f} "
        f"{result['pooled_ops_per_sec']:>15.2f} "
        f"{result['speedup']:>14.2f}x"
    )
    print(
        f"{'Latency (us)':<22} "
        f"{result['baseline_latency_us']:>15.4f} "
        f"{result['pooled_latency_us']:>15.4f} "
        f"{result['latency_delta_pct']:>+13.1f}%"
    )
    print("-" * 74)
    print(
        "Pool Stats: "
        f"hits/misses={pool_stats['cache_hits']}/{pool_stats['cache_misses']}, "
        f"sys_allocs={pool_stats['system_allocations']}, "
        f"cached={format_bytes(pool_stats['cached_bytes'])}"
    )
    print()


def print_summary(results):
    print("=" * 116)
    print("Allocator-Only Summary")
    print("=" * 116)
    print(
        f"{'Workload':<18} {'Domain':<10} {'Alloc':>10} {'Base ops/s':>12} "
        f"{'Pool ops/s':>12} {'Speedup':>10} {'Base us':>10} {'Pool us':>10} {'Delta':>10}"
    )
    print("-" * 116)
    for item, result in results:
        print(
            f"{item['name']:<18} "
            f"{item['domain']:<10} "
            f"{format_bytes(item['size_bytes']):>10} "
            f"{result['baseline_ops_per_sec']:>12.2f} "
            f"{result['pooled_ops_per_sec']:>12.2f} "
            f"{result['speedup']:>9.2f}x "
            f"{result['baseline_latency_us']:>10.4f} "
            f"{result['pooled_latency_us']:>10.4f} "
            f"{result['latency_delta_pct']:>+9.1f}%"
        )
    print()


def main():
    results = []
    for item in WORKLOADS:
        result = ap.benchmark_allocator(
            size_bytes=item["size_bytes"],
            iterations=item["iterations"],
            warmup=item["warmup"],
            domain=item["domain"],
        )
        print_report(item, result)
        results.append((item, result))
    print_summary(results)


if __name__ == "__main__":
    main()
