import gc

import asnumpy as ap


FRAG_KEYS = [
    "active_bytes",
    "cached_bytes",
    "internal_fragmentation_bytes",
    "internal_fragmentation_ratio_pct",
    "external_fragmentation_bytes",
    "external_fragmentation_ratio_pct",
    "largest_free_run_bytes",
    "largest_free_block_bytes",
    "run_utilization_pct",
    "hot_retained_empty_runs",
]

REENTRY_KEYS = [
    "system_allocations",
    "system_frees",
    "cache_hits",
    "cache_misses",
]


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def format_value(key, value):
    if key.endswith("_pct"):
        return f"{value:.2f}%"
    if "bytes" in key:
        return format_bytes(value)
    return str(value)


def checked(label, fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    if result is None:
        raise RuntimeError(f"{label} returned None; upstream exception was likely swallowed")
    return result


def zeros(shape, dtype="float32"):
    return checked(f"zeros(shape={shape}, dtype={dtype})", ap.zeros, shape, dtype=dtype)


def ones(shape, dtype="float32"):
    return checked(f"ones(shape={shape}, dtype={dtype})", ap.ones, shape, dtype=dtype)


def empty(shape, dtype="float32"):
    return checked(f"empty(shape={shape}, dtype={dtype})", ap.empty, shape, dtype=dtype)


def reduce_sum(arr, axis=0, keepdims=False):
    return checked(
        f"sum(shape={getattr(arr, 'shape', None)}, axis={axis}, keepdims={keepdims})",
        ap.sum,
        arr,
        axis=axis,
        keepdims=keepdims,
    )


def print_snapshot(title, domains):
    include_reentry_stats = "post_trim_reentry" in title
    print("=" * 108)
    print(title)
    print("=" * 108)
    for domain in domains:
        stats = ap.memory_stats(domain)
        print(f"[{domain}]")
        for key in FRAG_KEYS:
            print(f"{key:<34} {format_value(key, stats[key])}")
        if include_reentry_stats:
            print("reentry_reuse")
            for key in REENTRY_KEYS:
                print(f"{key:<34} {format_value(key, stats[key])}")
        print()


def assert_reentry_reuse(label, domain, max_system_allocations):
    stats = ap.memory_stats(domain)
    if stats["system_allocations"] > max_system_allocations:
        raise AssertionError(
            f"{label}[{domain}] exceeded reentry allocation budget: "
            f"{stats['system_allocations']} > {max_system_allocations}"
        )
    if stats["cache_hits"] <= 0:
        raise AssertionError(f"{label}[{domain}] expected at least one cache hit during reentry")
    if stats["cache_hits"] < stats["cache_misses"]:
        raise AssertionError(
            f"{label}[{domain}] cache reuse regressed: "
            f"hits={stats['cache_hits']} misses={stats['cache_misses']}"
        )


def reset_domains(*domains):
    ap.clear_executor_cache()
    ap.reset_executor_stats()
    ap.clear_descriptor_cache()
    ap.reset_descriptor_stats()
    for domain in domains:
        ap.clear_cache(domain)
    gc.collect()
    for domain in domains:
        ap.reset_memory_stats(domain)
    gc.collect()


def reset_stats_only(*domains):
    for domain in domains:
        ap.reset_memory_stats(domain)
    gc.collect()


def run_tensor_mixed_runtime():
    domain = "tensor"
    reset_domains(domain)
    print_snapshot("tensor_mixed_runtime:init", [domain])

    phase_a = []
    phase_b = []
    base_shapes = [(128, 128), (192, 192), (256, 256), (384, 384), (512, 512)]
    for _ in range(12):
        for shape in base_shapes:
            phase_a.append(empty(shape, dtype="float32"))
            phase_b.append(zeros(shape, dtype="float32"))
    print_snapshot("tensor_mixed_runtime:after_phase_alloc", [domain])

    phase_a = [arr for idx, arr in enumerate(phase_a) if idx % 2 == 0]
    phase_b = [arr for idx, arr in enumerate(phase_b) if idx % 3 == 0]
    gc.collect()
    print_snapshot("tensor_mixed_runtime:after_partial_free", [domain])

    refill = []
    refill_shapes = [(160, 160), (224, 224), (320, 320), (448, 448)]
    for _ in range(10):
        for shape in refill_shapes:
            refill.append(empty(shape, dtype="float32"))
    print_snapshot("tensor_mixed_runtime:after_refill", [domain])

    del refill
    del phase_a
    del phase_b
    gc.collect()
    print_snapshot("tensor_mixed_runtime:after_full_free", [domain])

    ap.trim_cache(domain)
    gc.collect()
    print_snapshot("tensor_mixed_runtime:after_trim", [domain])

    reset_stats_only(domain)
    reentry = []
    for _ in range(6):
        for shape in base_shapes:
            reentry.append(empty(shape, dtype="float32"))
            reentry.append(zeros(shape, dtype="float32"))
    print_snapshot("tensor_mixed_runtime:post_trim_reentry", [domain])
    assert_reentry_reuse("tensor_mixed_runtime", domain, max_system_allocations=12)
    del reentry
    gc.collect()


def run_tensor_workspace_mixed():
    reset_domains("tensor", "workspace")
    print_snapshot("tensor_workspace_mixed:init", ["tensor", "workspace"])

    inputs = [
        ones((256, 256), dtype="float32"),
        ones((512, 512), dtype="float32"),
        ones((1024, 1024), dtype="float32"),
    ]
    tensors = []
    outputs = []

    for _ in range(12):
        tensors.append(zeros((256, 256), dtype="float32"))
        tensors.append(empty((512, 512), dtype="float32"))
        for arr in inputs:
            outputs.append(reduce_sum(arr, axis=0, keepdims=False))
    print_snapshot("tensor_workspace_mixed:after_ops", ["tensor", "workspace"])

    tensors = [arr for idx, arr in enumerate(tensors) if idx % 2 == 0]
    outputs = [arr for idx, arr in enumerate(outputs) if idx % 3 == 0]
    gc.collect()
    print_snapshot("tensor_workspace_mixed:after_partial_free", ["tensor", "workspace"])

    refill = []
    for _ in range(8):
        refill.append(zeros((384, 384), dtype="float32"))
        refill.append(reduce_sum(inputs[1], axis=0, keepdims=False))
    print_snapshot("tensor_workspace_mixed:after_refill", ["tensor", "workspace"])

    del refill
    del tensors
    del outputs
    del inputs
    gc.collect()
    print_snapshot("tensor_workspace_mixed:after_full_free", ["tensor", "workspace"])

    ap.trim_cache("workspace")
    ap.trim_cache("tensor")
    gc.collect()
    print_snapshot("tensor_workspace_mixed:after_trim", ["tensor", "workspace"])

    reset_stats_only("tensor", "workspace")
    reentry_inputs = [
        ones((256, 256), dtype="float32"),
        ones((512, 512), dtype="float32"),
    ]
    reentry = []
    for _ in range(6):
        reentry.append(zeros((256, 256), dtype="float32"))
        reentry.append(empty((512, 512), dtype="float32"))
        reentry.append(reduce_sum(reentry_inputs[1], axis=0, keepdims=False))
    print_snapshot("tensor_workspace_mixed:post_trim_reentry", ["tensor", "workspace"])
    assert_reentry_reuse("tensor_workspace_mixed", "tensor", max_system_allocations=4)
    assert_reentry_reuse("tensor_workspace_mixed", "workspace", max_system_allocations=4)
    del reentry
    del reentry_inputs
    gc.collect()


def run_phase_style_runtime():
    reset_domains("tensor", "workspace")
    print_snapshot("phase_style_runtime:init", ["tensor", "workspace"])

    inputs = [ones((768, 768), dtype="float32") for _ in range(2)]

    for phase in range(1, 4):
        phase_tensors = []
        phase_outputs = []
        for _ in range(10):
            phase_tensors.append(empty((256 * phase, 256), dtype="float32"))
            phase_tensors.append(zeros((128 * phase, 512), dtype="float32"))
            for arr in inputs:
                phase_outputs.append(reduce_sum(arr, axis=0, keepdims=False))
        print_snapshot(f"phase_style_runtime:phase_{phase}_active", ["tensor", "workspace"])

        del phase_tensors
        del phase_outputs
        gc.collect()
        print_snapshot(f"phase_style_runtime:phase_{phase}_released", ["tensor", "workspace"])

    del inputs
    gc.collect()
    ap.trim_cache("workspace")
    ap.trim_cache("tensor")
    gc.collect()
    print_snapshot("phase_style_runtime:after_trim", ["tensor", "workspace"])

    reset_stats_only("tensor", "workspace")
    reentry_inputs = [ones((768, 768), dtype="float32")]
    reentry_tensors = []
    reentry_outputs = []
    for _ in range(8):
        reentry_tensors.append(empty((256, 256), dtype="float32"))
        reentry_tensors.append(zeros((128, 512), dtype="float32"))
        for arr in reentry_inputs:
            reentry_outputs.append(reduce_sum(arr, axis=0, keepdims=False))
    print_snapshot("phase_style_runtime:post_trim_reentry", ["tensor", "workspace"])
    assert_reentry_reuse("phase_style_runtime", "tensor", max_system_allocations=6)
    assert_reentry_reuse("phase_style_runtime", "workspace", max_system_allocations=4)
    del reentry_tensors
    del reentry_outputs
    del reentry_inputs
    gc.collect()


def main():
    ap.init()
    ap.set_device(0)
    ap.refresh_memory_pool_config()

    workloads = [
        ("tensor_mixed_runtime", run_tensor_mixed_runtime),
        ("tensor_workspace_mixed", run_tensor_workspace_mixed),
        ("phase_style_runtime", run_phase_style_runtime),
    ]
    for name, fn in workloads:
        try:
            fn()
        except Exception as exc:
            raise RuntimeError(f"benchmark_fragmentation failed in {name}") from exc


if __name__ == "__main__":
    main()
