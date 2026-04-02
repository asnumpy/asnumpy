import asnumpy as ap


CASES = [
    {
        "name": "small_path",
        "shape": (256, 256),  # 256 KB float32
        "expected_counter": "small_runs",
    },
    {
        "name": "large_path",
        "shape": (1024, 1024),  # 4 MB float32
        "expected_counter": "large_arenas",
    },
]


def print_stats(tag):
    stats = ap.memory_stats()
    print(
        f"[{tag}] "
        f"active={stats['active_bytes']} "
        f"cached={stats['cached_bytes']} "
        f"system={stats['system_bytes']} "
        f"hits={stats['cache_hits']} "
        f"misses={stats['cache_misses']} "
        f"small_runs={stats['small_run_count']} "
        f"large_arenas={stats['large_arena_count']}"
    )


def run_case(case):
    print("=" * 60)
    print(
        f"Smoke test: {case['name']} | "
        f"shape={case['shape']} | expected={case['expected_counter']}"
    )
    print("=" * 60)

    ap.trim_cache()
    print_stats(f"{case['name']}:init")

    a = ap.zeros(case["shape"], dtype="float32")
    print_stats(f"{case['name']}:after_a")

    del a
    print_stats(f"{case['name']}:after_del_a")

    b = ap.zeros(case["shape"], dtype="float32")
    print_stats(f"{case['name']}:after_b")

    del b
    ap.trim_cache()
    print_stats(f"{case['name']}:after_trim")
    print()


def main():
    for case in CASES:
        run_case(case)

    print("=" * 60)
    print("Expected checks:")
    print("1. small_path should make `small_runs` rise above 0 while it is active/cached.")
    print("2. large_path should make `large_arenas` rise above 0 while it is active/cached.")
    print("3. The second allocation in each case should increase `cache_hits`.")
    print("4. `trim_cache()` should bring cached/system bytes back down when fully idle.")
    print("=" * 60)


if __name__ == "__main__":
    main()
