"""VMM Stitched Large Allocator integration tests.

Covers:
  T1   - Large stitched alloc/free (>1MB), single size
  T1b  - Multi-size stitched alloc/free (all >1MB, across chunk boundaries)
  T2   - free -> clear_cache -> decompose (verify segment_count drops to 0)
  T2b  - Free stitched block reuse (same-size realloc, stitched_reuse_hits)
  T2c  - Decompose -> chunk cache reuse (force decompose via clear_cache,
         then alloc different size using recycled chunks)
  T3   - ASN_POOL_VMM_ENABLE=0 legacy fallback
  T4   - Workspace domain via AclWorkspace RAII
  T5   - Small blocks (<1MB) unaffected
  T5b  - Mixed small/large interleaving
  T7   - shutdown_runtime unified L3->L2->L1 cleanup
  T7b  - Clean exit with cached large stitched blocks
  Stats - VMM stats fields present
"""
import os
import sys
import subprocess
import textwrap

PYTHON = sys.executable


def _run(script_text, env_override=None, timeout=30):
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = subprocess.run(
        [PYTHON, "-c", textwrap.dedent(script_text)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


# ---------------------------------------------------------------------------
# T1: Basic stitched alloc/free
# ---------------------------------------------------------------------------
def test_t1_stitched_alloc_free():
    """T1: Large stitched alloc/free (>1MB) with tensor-domain stats."""
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((512, 1024), dtype='float32')  # 2MB, clearly >1MB threshold
        ts = ap.memory_stats('tensor')
        assert ts['active_bytes'] > 0, f"tensor active_bytes should be >0, got {ts['active_bytes']}"
        print(f"  alloc 2MB: tensor active_bytes={ts['active_bytes']}")

        del a
        ts2 = ap.memory_stats('tensor')
        assert ts2['active_bytes'] == 0, \\
            f"tensor active_bytes should be 0 after free, got {ts2['active_bytes']}"
        print("T1 PASS")
    """)
    print(f"T1 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T1 stderr: {r.stderr}")
    assert r.returncode == 0, f"T1 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T1b: Multi-size stitched (all sizes > 1MB)
# ---------------------------------------------------------------------------
def test_t1_multi_size_stitched():
    """T1b: Multiple large sizes that are all >1MB, crossing chunk boundaries."""
    r = _run("""
        import asnumpy as ap

        # All sizes are strictly >1MB (kSmallAllocationMax = 1MB = 1048576)
        # chunk = 2MB, so these cross 1, 2, 4 chunk boundaries
        sizes = [
            (512, 1024),   # 2MB  = 1 chunk
            (1024, 1024),  # 4MB  = 2 chunks
            (2048, 1024),  # 8MB  = 4 chunks
            (512, 4096),   # 8MB  = 4 chunks
            (4096, 1024),  # 16MB = 8 chunks
        ]
        tensors = []
        for h, w in sizes:
            tensors.append(ap.zeros((h, w), dtype='float32'))
            ts = ap.memory_stats('tensor')
            byte_size = h * w * 4
            print(f"  alloc ({h}x{w})={byte_size}B: active_bytes={ts['active_bytes']}")

        ts_before = ap.memory_stats('tensor')
        total_expected = sum(h * w * 4 for h, w in sizes)
        assert ts_before['active_bytes'] >= total_expected, \\
            f"active_bytes {ts_before['active_bytes']} < expected {total_expected}"

        tensors.clear()

        ts_after = ap.memory_stats('tensor')
        assert ts_after['active_bytes'] == 0, \\
            f"tensor active_bytes should be 0 after clear, got {ts_after['active_bytes']}"
        print("T1b PASS")
    """)
    print(f"T1b stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T1b stderr: {r.stderr}")
    assert r.returncode == 0, f"T1b failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T2: free -> clear_cache -> decompose (strict)
# ---------------------------------------------------------------------------
def test_t2_clear_cache_decompose():
    """T2: free -> clear_cache verifies decompose via stitched_segment_count."""
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype='float32')  # 4MB
        b = ap.zeros((2048, 1024), dtype='float32')  # 8MB
        del a
        del b

        ts_before = ap.memory_stats('tensor')
        cached_before = ts_before['cached_bytes']
        seg_before = ts_before['stitched_segment_count']
        print(f"  before clear_cache: cached={cached_before}, segments={seg_before}")
        assert cached_before > 0, f"cached_bytes should be >0 after free, got {cached_before}"
        assert seg_before > 0, f"stitched_segment_count should be >0, got {seg_before}"

        ap.clear_cache()

        ts_after = ap.memory_stats('tensor')
        cached_after = ts_after['cached_bytes']
        seg_after = ts_after['stitched_segment_count']
        print(f"  after clear_cache: cached={cached_after}, segments={seg_after}")
        assert cached_after == 0, \\
            f"cached_bytes should be exactly 0 after clear_cache, got {cached_after}"
        assert seg_after == 0, \\
            f"stitched_segment_count should be 0 after decompose, got {seg_after}"
        print("T2 PASS")
    """)
    print(f"T2 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T2 stderr: {r.stderr}")
    assert r.returncode == 0, f"T2 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T2b: Free stitched block reuse (sPool hit, stitched_reuse_hits)
# ---------------------------------------------------------------------------
def test_t2b_stitched_reuse():
    """T2b: Free a stitched block, realloc same size -> sPool reuse."""
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype='float32')  # 4MB
        del a
        # Block is now in dirty_stitched_blocks free list

        ts1 = ap.memory_stats('tensor')
        hits_before = ts1['stitched_reuse_hits']

        b = ap.zeros((1024, 1024), dtype='float32')  # same 4MB -> should hit sPool
        ts2 = ap.memory_stats('tensor')
        hits_after = ts2['stitched_reuse_hits']
        print(f"  reuse_hits: before={hits_before}, after={hits_after}")
        assert hits_after > hits_before, \\
            f"stitched_reuse_hits should increase: before={hits_before} after={hits_after}"

        del b
        print("T2b PASS")
    """)
    print(f"T2b stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T2b stderr: {r.stderr}")
    assert r.returncode == 0, f"T2b failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T2c: Decompose -> chunk cache -> new segment reuse
# ---------------------------------------------------------------------------
def test_t2c_decompose_chunk_reuse():
    """T2c: sPool has only smaller blocks; allocating larger size forces
    decompose_free_stitched_blocks_for_request_locked, which decomposes
    the free blocks into chunk cache, then borrows chunks to build the
    new segment.

    Each zeros() call creates 2 tensor-domain large allocs (C++ original +
    pybind11 copy). To isolate chunk reuse, we reset_memory_stats after
    seeding the sPool, then observe that the new 8MB alloc needs far fewer
    physical allocations than the naive 8 chunks (4 per alloc × 2 allocs).

    Scenario:
      1. Alloc two 4MB blocks, free both -> sPool holds 4MB blocks
      2. reset_memory_stats (counters zeroed, pool state preserved)
      3. Alloc 8MB -> sPool miss (4MB < 8MB)
         -> decompose free 4MB blocks -> borrow chunks from cache
         -> fewer new physical allocations than naive expectation
    """
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype='float32')  # 4MB
        b = ap.zeros((1024, 1024), dtype='float32')  # 4MB
        del a
        del b

        ts_seed = ap.memory_stats('tensor')
        print(f"  seeded sPool: cached_bytes={ts_seed['cached_bytes']}, "
              f"segments={ts_seed['stitched_segment_count']}")

        ap.reset_memory_stats('tensor')

        c = ap.zeros((2048, 1024), dtype='float32')  # 8MB = 4 chunks

        ts = ap.memory_stats('tensor', verbose=True)
        new_sys_allocs = ts['system_allocations']
        # 8MB = 4 chunks. Without reuse, each alloc (C++ + pybind copy) would
        # need 4 fresh chunks, totaling 8 new physical allocs.
        naive_expected = 8
        print(f"  8MB alloc: new system_allocations={new_sys_allocs}, "
              f"naive_expected={naive_expected}")

        assert new_sys_allocs < naive_expected, \\
            f"system_allocations ({new_sys_allocs}) should be < {naive_expected} " \\
            f"if chunk cache reuse from decompose worked"

        del c
        print("T2c PASS")
    """)
    print(f"T2c stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T2c stderr: {r.stderr}")
    assert r.returncode == 0, f"T2c failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T3: VMM disabled fallback
# ---------------------------------------------------------------------------
def test_t3_vmm_disabled_fallback():
    """T3: ASN_POOL_VMM_ENABLE=0 forces legacy fallback."""
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype='float32')  # 4MB via legacy
        ts = ap.memory_stats('tensor')
        fallbacks = ts['vmm_fallback_allocations']
        print(f"  VMM disabled: active_bytes={ts['active_bytes']}, vmm_fallback={fallbacks}")
        assert ts['active_bytes'] > 0, \\
            f"allocation should still work via legacy, got active_bytes={ts['active_bytes']}"
        assert fallbacks > 0, \\
            f"vmm_fallback_allocations should be >0 with VMM disabled, got {fallbacks}"

        del a
        ts2 = ap.memory_stats('tensor')
        assert ts2['active_bytes'] == 0, \\
            f"active_bytes should be 0 after free, got {ts2['active_bytes']}"
        print("T3 PASS")
    """, env_override={"ASN_POOL_VMM_ENABLE": "0"})
    print(f"T3 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T3 stderr: {r.stderr}")
    assert r.returncode == 0, f"T3 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T4: Workspace domain
# ---------------------------------------------------------------------------
def test_t4_workspace_domain():
    """T4: Workspace domain allocation via AclWorkspace RAII.

    AclWorkspace sizes are determined by CANN kernels. zeros/ones may
    request workspaceSize=0, but sum/reduce always request >0 workspace.
    We use ap.sum() to trigger a workspace allocation, then assert the
    workspace domain allocation_requests counter increased.
    """
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((512, 1024), dtype='float32')  # 2MB tensor

        ws_before = ap.memory_stats('workspace', verbose=True)
        alloc_before = ws_before['allocation_requests']
        print(f"  workspace alloc_requests before sum: {alloc_before}")

        # ap.sum() triggers aclnnReduceSumGetWorkspaceSize -> AclWorkspace(size>0)
        s = ap.sum(a)

        ws_after = ap.memory_stats('workspace', verbose=True)
        alloc_after = ws_after['allocation_requests']
        free_after = ws_after['free_requests']
        print(f"  workspace alloc_requests after sum: {alloc_after}")
        print(f"  workspace free_requests: {free_after}")

        assert alloc_after > alloc_before, \\
            f"workspace allocation_requests should increase after sum: " \\
            f"before={alloc_before} after={alloc_after}"

        # Workspace is RAII, freed immediately, so active_bytes should be 0
        assert ws_after['active_bytes'] == 0, \\
            f"workspace active_bytes should be 0 (RAII freed), got {ws_after['active_bytes']}"

        # Tensor domain is independent
        ts = ap.memory_stats('tensor')
        assert ts['active_bytes'] > 0, "tensor domain should have active allocations"
        print(f"  tensor active_bytes: {ts['active_bytes']}")

        del a
        del s
        print("T4 PASS")
    """)
    print(f"T4 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T4 stderr: {r.stderr}")
    assert r.returncode == 0, f"T4 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T5: Small blocks
# ---------------------------------------------------------------------------
def test_t5_small_blocks():
    """T5: Small blocks (<1MB) use small allocator, unaffected by VMM."""
    r = _run("""
        import asnumpy as ap

        small_sizes = [
            (1, 1),       # 4B
            (16, 16),     # 1KB
            (128, 128),   # 64KB
            (256, 256),   # 256KB
            (256, 512),   # 512KB
        ]
        tensors = []
        for h, w in small_sizes:
            tensors.append(ap.zeros((h, w), dtype='float32'))

        ts = ap.memory_stats('tensor')
        print(f"  small blocks: active_bytes={ts['active_bytes']}")
        assert ts['active_bytes'] > 0, "small allocations should work"

        tensors.clear()

        ts2 = ap.memory_stats('tensor')
        print(f"  after free: active_bytes={ts2['active_bytes']}")
        assert ts2['active_bytes'] == 0, \\
            f"active_bytes should be 0 after all small frees, got {ts2['active_bytes']}"
        print("T5 PASS")
    """)
    print(f"T5 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T5 stderr: {r.stderr}")
    assert r.returncode == 0, f"T5 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T5b: Mixed small/large
# ---------------------------------------------------------------------------
def test_t5_mixed_small_large():
    """T5b: Mixed small and large allocation interleaving."""
    r = _run("""
        import asnumpy as ap

        small = ap.zeros((128, 128), dtype='float32')   # 64KB  - small path
        large = ap.zeros((1024, 1024), dtype='float32')  # 4MB   - stitched path
        small2 = ap.zeros((64, 64), dtype='float32')     # 16KB  - small path
        large2 = ap.zeros((512, 1024), dtype='float32')  # 2MB   - stitched path

        ts = ap.memory_stats('tensor')
        print(f"  mixed alloc: active_bytes={ts['active_bytes']}")
        assert ts['active_bytes'] > 0

        del large
        del small
        del large2
        del small2

        ts2 = ap.memory_stats('tensor')
        assert ts2['active_bytes'] == 0, \\
            f"active_bytes should be 0 after mixed frees, got {ts2['active_bytes']}"
        print("T5b PASS")
    """)
    print(f"T5b stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T5b stderr: {r.stderr}")
    assert r.returncode == 0, f"T5b failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T7: shutdown_runtime (L3 -> L2 -> L1)
# ---------------------------------------------------------------------------
def test_t7_shutdown_runtime():
    """T7: shutdown_runtime clears all three cache layers."""
    r = _run("""
        import asnumpy as ap

        # Trigger allocations that populate L1 (memory), L2 (descriptor), L3 (executor)
        a = ap.zeros((1024, 1024), dtype='float32')
        b = ap.zeros((128, 128), dtype='float32')
        del a
        del b

        ts_before = ap.memory_stats('tensor')
        print(f"  before shutdown: cached_bytes={ts_before['cached_bytes']}")
        assert ts_before['cached_bytes'] > 0, "should have cached bytes before shutdown"

        ap.shutdown_runtime()

        ts_after = ap.memory_stats('tensor')
        cached = ts_after['cached_bytes']
        print(f"  after shutdown: cached_bytes={cached}")
        assert cached == 0, f"cached_bytes should be 0 after shutdown_runtime, got {cached}"

        # Verify L2/L3 caches are also cleared
        ds = ap.descriptor_stats()
        es = ap.executor_stats()
        print(f"  descriptor cached_count={ds['cached_count']}, executor cached_count={es['cached_count']}")
        assert ds['cached_count'] == 0, \\
            f"descriptor cache should be empty after shutdown, got {ds['cached_count']}"
        assert es['cached_count'] == 0, \\
            f"executor cache should be empty after shutdown, got {es['cached_count']}"
        print("T7 PASS")
    """)
    print(f"T7 stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T7 stderr: {r.stderr}")
    assert r.returncode == 0, f"T7 failed: {r.stderr}"


# ---------------------------------------------------------------------------
# T7b: Clean exit with cached large stitched blocks (regression for exit crash)
# ---------------------------------------------------------------------------
def test_t7_clean_exit():
    """T7b: Alloc large -> free -> exit with cached stitched blocks.

    Regression for the aclrtFreePhysical terminate-on-exit crash.
    The key scenario is: VMM resources are cached when atexit runs.
    """
    r = _run("""
        import asnumpy as ap

        # Alloc large stitched blocks
        a = ap.zeros((1024, 1024), dtype='float32')   # 4MB
        b = ap.zeros((2048, 1024), dtype='float32')    # 8MB

        # Free them -> they become cached stitched blocks (dirty/retained)
        del a
        del b

        ts = ap.memory_stats('tensor')
        assert ts['cached_bytes'] > 0, \\
            f"should have cached stitched blocks before exit, got {ts['cached_bytes']}"
        print(f"  exiting with cached_bytes={ts['cached_bytes']}, "
              f"segments={ts['stitched_segment_count']}")
        print("T7b PASS")
    """, timeout=15)
    print(f"T7b stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"T7b stderr: {r.stderr}")
    assert "terminate called" not in r.stderr, f"Exit crash detected: {r.stderr}"
    assert "aclrtFreePhysical" not in r.stderr, f"Physical free error on exit: {r.stderr}"
    assert r.returncode == 0, f"T7b exit code {r.returncode}: {r.stderr}"


# ---------------------------------------------------------------------------
# Stats: VMM fields present
# ---------------------------------------------------------------------------
def test_vmm_stats_fields():
    """Verify all VMM-related stats fields are present and typed correctly."""
    r = _run("""
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype='float32')
        ts = ap.memory_stats('tensor')

        required_fields = [
            'vmm_internal_fragmentation_bytes',
            'stitched_reuse_hits',
            'vmm_fallback_allocations',
            'chunk_cache_bytes',
            'chunk_cache_count',
            'stitched_segment_count',
        ]
        for field in required_fields:
            assert field in ts, f"Missing tensor stats field: {field}"
            assert isinstance(ts[field], int), f"{field} should be int, got {type(ts[field])}"
            print(f"  {field} = {ts[field]}")

        del a
        print("Stats PASS")
    """)
    print(f"Stats stdout: {r.stdout}")
    if r.returncode != 0:
        print(f"Stats stderr: {r.stderr}")
    assert r.returncode == 0, f"Stats test failed: {r.stderr}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("T1   - Stitched alloc/free", test_t1_stitched_alloc_free),
        ("T1b  - Multi-size stitched", test_t1_multi_size_stitched),
        ("T2   - clear_cache decompose", test_t2_clear_cache_decompose),
        ("T2b  - Stitched sPool reuse", test_t2b_stitched_reuse),
        ("T2c  - Decompose chunk reuse", test_t2c_decompose_chunk_reuse),
        ("T3   - VMM disabled fallback", test_t3_vmm_disabled_fallback),
        ("T4   - Workspace domain", test_t4_workspace_domain),
        ("T5   - Small blocks", test_t5_small_blocks),
        ("T5b  - Mixed small/large", test_t5_mixed_small_large),
        ("T7   - shutdown_runtime", test_t7_shutdown_runtime),
        ("T7b  - Clean exit (cached)", test_t7_clean_exit),
        ("Stats - VMM fields", test_vmm_stats_fields),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            func()
            passed += 1
            print(f">>> {name}: PASSED")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f">>> {name}: FAILED - {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err[:300]}")
    sys.exit(0 if failed == 0 else 1)
