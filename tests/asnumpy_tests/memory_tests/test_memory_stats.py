import pytest

import asnumpy as ap


@pytest.mark.parametrize("domain", [None, "tensor", "workspace"])
def test_memory_stats_exposes_vmm_fields(domain):
    stats = ap.memory_stats(domain=domain, verbose=True)

    expected_keys = {
        "vmm_internal_fragmentation_bytes",
        "stitched_reuse_hits",
        "vmm_fallback_allocations",
        "chunk_cache_bytes",
        "chunk_cache_count",
        "stitched_segment_count",
    }

    missing = expected_keys.difference(stats)
    assert not missing, f"missing memory stats keys: {sorted(missing)}"


def test_shutdown_runtime_is_exposed_and_callable():
    ap.shutdown_runtime()
