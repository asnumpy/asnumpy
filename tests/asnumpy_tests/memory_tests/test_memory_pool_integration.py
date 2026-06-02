import subprocess
import sys
import textwrap


def _run_asnumpy_script(script_text, env_override=None, timeout=90):
    env = None
    if env_override:
        import os

        env = os.environ.copy()
        env.update(env_override)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script_text)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _assert_success(result):
    message = (
        f"subprocess failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.returncode == 0, message


def test_creation_module_small_tensor_path():
    result = _run_asnumpy_script(
        """
        import gc
        import numpy as np
        import asnumpy as ap

        ap.clear_cache()
        ap.reset_memory_stats("tensor")

        a = ap.zeros((128, 128), dtype="float32")
        np.testing.assert_array_equal(a.to_numpy(), np.zeros((128, 128), dtype=np.float32))

        stats = ap.memory_stats("tensor", verbose=True)
        assert stats["allocation_requests"] > 0, stats
        assert stats["active_bytes"] > 0, stats
        assert stats["stitched_segment_count"] == 0, stats

        del a
        gc.collect()

        stats_after = ap.memory_stats("tensor", verbose=True)
        assert stats_after["active_bytes"] == 0, stats_after
        print("creation small path ok")
        """
    )
    _assert_success(result)


def test_creation_module_large_tensor_path():
    result = _run_asnumpy_script(
        """
        import gc
        import numpy as np
        import asnumpy as ap

        ap.clear_cache()
        ap.reset_memory_stats("tensor")

        a = ap.zeros((1024, 1024), dtype="float32")
        np.testing.assert_array_equal(a.to_numpy(), np.zeros((1024, 1024), dtype=np.float32))

        stats = ap.memory_stats("tensor", verbose=True)
        assert stats["allocation_requests"] > 0, stats
        assert stats["active_bytes"] > 0, stats
        if stats["vmm_fallback_allocations"] == 0:
            assert stats["stitched_segment_count"] >= 1, stats

        del a
        gc.collect()

        stats_after = ap.memory_stats("tensor", verbose=True)
        assert stats_after["active_bytes"] == 0, stats_after
        print("creation large path ok")
        """
    )
    _assert_success(result)


def test_math_sum_workspace_path():
    result = _run_asnumpy_script(
        """
        import gc
        import numpy as np
        import asnumpy as ap

        a = ap.ones((1024, 1024), dtype="float32")
        ap.reset_memory_stats("workspace")

        out = ap.sum(a, axis=0, keepdims=False)
        np.testing.assert_allclose(
            out.to_numpy(),
            np.sum(np.ones((1024, 1024), dtype=np.float32), axis=0),
            rtol=1e-5,
            atol=1e-5,
        )

        stats = ap.memory_stats("workspace", verbose=True)
        assert stats["allocation_requests"] > 0, stats
        assert stats["active_bytes"] == 0, stats

        del out
        del a
        gc.collect()
        print("sum workspace path ok")
        """
    )
    _assert_success(result)


def test_sorting_module_workspace_path():
    result = _run_asnumpy_script(
        """
        import gc
        import numpy as np
        import asnumpy as ap

        data = np.tile(np.arange(1024, 0, -1, dtype=np.float32), (1024, 1))
        a = ap.ndarray.from_numpy(data)
        ap.reset_memory_stats("workspace")

        out = ap.sort(a, axis=-1)
        np.testing.assert_allclose(out.to_numpy(), np.sort(data, axis=-1), rtol=1e-5, atol=1e-5)

        stats = ap.memory_stats("workspace", verbose=True)
        assert stats["allocation_requests"] > 0, stats
        assert stats["active_bytes"] == 0, stats

        del out
        del a
        gc.collect()
        print("sort workspace path ok")
        """
    )
    _assert_success(result)


def test_statistics_mean_workspace_path():
    result = _run_asnumpy_script(
        """
        import gc
        import numpy as np
        import asnumpy as ap

        data = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
        a = ap.ndarray.from_numpy(data)
        ap.reset_memory_stats("workspace")

        out = ap.mean(a, axis=0, keepdims=False)
        np.testing.assert_allclose(out.to_numpy(), np.mean(data, axis=0), rtol=1e-5, atol=1e-5)

        stats = ap.memory_stats("workspace", verbose=True)
        assert stats["allocation_requests"] > 0, stats
        assert stats["active_bytes"] == 0, stats

        del out
        del a
        gc.collect()
        print("mean workspace path ok")
        """
    )
    _assert_success(result)


def test_clear_cache_releases_cached_tensor_blocks():
    result = _run_asnumpy_script(
        """
        import gc
        import asnumpy as ap

        ap.clear_cache("tensor")
        ap.reset_memory_stats("tensor")

        a = ap.zeros((1024, 1024), dtype="float32")
        b = ap.zeros((2048, 1024), dtype="float32")
        del a
        del b
        gc.collect()

        stats_before = ap.memory_stats("tensor", verbose=True)
        assert stats_before["cached_bytes"] > 0, stats_before

        ap.clear_cache("tensor")

        stats_after = ap.memory_stats("tensor", verbose=True)
        assert stats_after["cached_bytes"] == 0, stats_after
        assert stats_after["chunk_cache_count"] == 0, stats_after
        assert stats_after["stitched_segment_count"] == 0, stats_after
        print("clear_cache cleanup ok")
        """
    )
    _assert_success(result)


def test_vmm_disabled_falls_back_for_large_tensor_allocations():
    result = _run_asnumpy_script(
        """
        import gc
        import asnumpy as ap

        ap.clear_cache("tensor")
        ap.reset_memory_stats("tensor")

        a = ap.zeros((1024, 1024), dtype="float32")
        stats = ap.memory_stats("tensor", verbose=True)
        assert stats["active_bytes"] > 0, stats
        assert stats["vmm_fallback_allocations"] >= 1, stats

        del a
        gc.collect()

        stats_after = ap.memory_stats("tensor", verbose=True)
        assert stats_after["active_bytes"] == 0, stats_after
        print("legacy fallback ok")
        """,
        env_override={"ASN_POOL_VMM_ENABLE": "0"},
    )
    _assert_success(result)


def test_shutdown_runtime_clears_cached_layers():
    result = _run_asnumpy_script(
        """
        import gc
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype="float32")
        b = ap.sum(a, axis=0, keepdims=False)
        del a
        del b
        gc.collect()

        ap.shutdown_runtime()

        tensor_stats = ap.memory_stats("tensor", verbose=True)
        workspace_stats = ap.memory_stats("workspace", verbose=True)
        descriptor_stats = ap.descriptor_stats()
        executor_stats = ap.executor_stats()

        assert tensor_stats["cached_bytes"] == 0, tensor_stats
        assert tensor_stats["stitched_segment_count"] == 0, tensor_stats
        assert workspace_stats["cached_bytes"] == 0, workspace_stats
        assert workspace_stats["stitched_segment_count"] == 0, workspace_stats
        assert descriptor_stats["cached_count"] == 0, descriptor_stats
        assert executor_stats["cached_count"] == 0, executor_stats
        print("shutdown runtime cleanup ok")
        """
    )
    _assert_success(result)


def test_process_exits_cleanly_after_cached_large_blocks():
    result = _run_asnumpy_script(
        """
        import gc
        import asnumpy as ap

        a = ap.zeros((1024, 1024), dtype="float32")
        b = ap.zeros((2048, 1024), dtype="float32")
        del a
        del b
        gc.collect()

        stats = ap.memory_stats("tensor", verbose=True)
        assert stats["cached_bytes"] > 0, stats
        print("prepared cached large blocks for exit")
        """
    )
    _assert_success(result)
    assert "terminate called" not in result.stderr, result.stderr
    assert "aclrtFreePhysical failed" not in result.stderr, result.stderr
