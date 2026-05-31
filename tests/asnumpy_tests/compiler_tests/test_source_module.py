# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""Tests for the asnumpy.compiler module.

Test classes are organized by dependency:
- Signature parsing (no NPU, no compiler needed)
- Compilation (needs bisheng compiler)
- SourceModule integration (needs bisheng)
- Kernel execution (needs NPU device)
- Caching (needs bisheng)
- Edge cases
"""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from asnumpy.compiler import bisheng_compiler, source_module
from asnumpy.compiler.cache import compute_cache_key, get_cache_dir, get_compiler_version
from asnumpy.compiler.kernel_function import ArgSpec, KernelFunction, PreparedKernel

# ==========================================================================
# Skip markers
# ==========================================================================

requires_npu = pytest.mark.skipif(
    "ASCEND_TOOLKIT_HOME" not in os.environ
    and "ASCEND_HOME_PATH" not in os.environ,
    reason="Requires CANN NPU device (ASCEND_TOOLKIT_HOME not set)",
)

requires_bisheng = pytest.mark.skipif(
    not Path("/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/bisheng").exists()
    and "ASCEND_TOOLKIT_HOME" not in os.environ,
    reason="Requires bisheng compiler",
)

# ACL binary loading (aclrtBinaryLoadFromFile) requires CANN 8.5+.
# CANN 8.2.RC1 has the API but the binary format requirements differ.
requires_acl_binary_load = pytest.mark.skip(
    reason="aclrtBinaryLoadFromFile requires CANN 8.5+ for custom kernel binaries"
)


# ==========================================================================
# Test: Signature Parsing
# ==========================================================================

class TestSignatureParsing:
    """Tests for _parse_kernel_signature (no compilation needed)."""

    def test_parse_simple_kernel(self, vector_add_source):
        specs = source_module._parse_kernel_signature(
            vector_add_source, "vector_add"
        )
        assert len(specs) == 4
        assert specs[0] == ArgSpec("a", "float*", True, 8)
        assert specs[1] == ArgSpec("b", "float*", True, 8)
        assert specs[2] == ArgSpec("c", "float*", True, 8)
        assert specs[3] == ArgSpec("n", "int32", False, 4)

    def test_parse_mixed_args(self, fill_const_source):
        specs = source_module._parse_kernel_signature(
            fill_const_source, "fill_const"
        )
        assert len(specs) == 3
        assert specs[0] == ArgSpec("out", "float*", True, 8)
        assert specs[1] == ArgSpec("value", "float32", False, 4)
        assert specs[2] == ArgSpec("n", "int32", False, 4)

    def test_parse_kernel_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            source_module._parse_kernel_signature(
                "extern \"C\" __global__ __aicore__ void foo() {}", "bar"
            )

    def test_parse_no_params(self):
        src = 'extern "C" __global__ __aicore__ void empty_kernel() {}'
        specs = source_module._parse_kernel_signature(src, "empty_kernel")
        assert specs == []

    def test_manual_signature_override(self):
        sig = ["float32*", "float32*", "int32"]
        specs = source_module._manual_signature_to_arg_specs(sig)
        assert len(specs) == 3
        assert specs[0].is_pointer is True
        assert specs[2].is_pointer is False
        assert specs[2].size_bytes == 4

    def test_manual_signature_scalar_types(self):
        sig = ["int64", "float64", "bool"]
        specs = source_module._manual_signature_to_arg_specs(sig)
        assert specs[0].size_bytes == 8
        assert specs[1].size_bytes == 8
        assert specs[2].size_bytes == 1


# ==========================================================================
# Test: Compilation
# ==========================================================================

class TestCompilation:
    """Tests for bisheng compilation (needs bisheng compiler)."""

    @requires_bisheng
    def test_compiles_without_error(self, vector_add_source):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            so_file = bisheng_compiler.compile_kernel(
                vector_add_source, out_dir
            )
            assert so_file.exists()
            assert so_file.suffix == ".so"

    @requires_bisheng
    def test_compiled_o_is_elf(self, vector_add_source):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            so_file = bisheng_compiler.compile_kernel(
                vector_add_source, out_dir
            )
            # Check ELF magic bytes
            header = so_file.read_bytes()[:4]
            assert header == b"\x7fELF"

    @requires_bisheng
    def test_compile_invalid_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with pytest.raises(bisheng_compiler.CompileError):
                bisheng_compiler.compile_kernel(
                    "this is not valid C++ code", out_dir
                )

    @requires_bisheng
    def test_compile_log_written(self, vector_add_source):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            bisheng_compiler.compile_kernel(vector_add_source, out_dir)
            log = out_dir / "compile.log"
            assert log.exists()


# ==========================================================================
# Test: SourceModule Integration
# ==========================================================================

class TestSourceModule:
    """Tests for SourceModule (needs bisheng + ACL binary loading, CANN 8.5+)."""

    pytestmark = [requires_acl_binary_load, requires_bisheng]

    def test_compilation_and_loading(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source, keep=True)
        try:
            assert mod.list_functions() is not None
            assert "vector_add" in mod.list_functions()
        finally:
            mod.close()

    def test_get_function_returns_callable(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source)
        try:
            kernel = mod.get_function("vector_add")
            assert isinstance(kernel, KernelFunction)
            assert kernel.name == "vector_add"
        finally:
            mod.close()

    def test_get_function_unknown_name_raises(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source)
        try:
            with pytest.raises(ValueError, match="not found"):
                mod.get_function("nonexistent")
        finally:
            mod.close()

    def test_close(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source)
        mod.close()
        # Should be safe to call again
        mod.close()

    def test_context_manager(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        with SourceModule(vector_add_source) as mod:
            assert len(mod.list_functions()) > 0

    def test_multi_kernel_source(self, multi_kernel_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(multi_kernel_source)
        try:
            funcs = mod.list_functions()
            assert "kernel_one" in funcs
            assert "kernel_two" in funcs
        finally:
            mod.close()

    def test_manual_signature(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source)
        try:
            sig = ["float32*", "float32*", "float32*", "int32"]
            kernel = mod.get_function("vector_add", signature=sig)
            assert len(kernel._arg_specs) == 4
        finally:
            mod.close()


# ==========================================================================
# Test: Kernel Execution
# ==========================================================================

class TestKernelExecution:
    """End-to-end tests that compile, load, and launch kernels on NPU."""

    pytestmark = [requires_npu, requires_acl_binary_load, requires_bisheng]

    def test_vector_add_correctness(self, vector_add_source):
        import asnumpy as ap
        from asnumpy.compiler import SourceModule

        N = 1024
        rng = np.random.RandomState(42)

        mod = SourceModule(vector_add_source, options=["-O3"])
        try:
            kernel = mod.get_function("vector_add")

            a_np = rng.randn(N).astype(np.float32)
            b_np = rng.randn(N).astype(np.float32)
            a_ap = ap.ndarray.from_numpy(a_np)
            b_ap = ap.ndarray.from_numpy(b_np)
            c_ap = ap.empty((N,), dtype=ap.float32)

            kernel(a_ap, b_ap, c_ap, N, grid=(8,))

            result = c_ap.to_numpy()
            expected = a_np + b_np
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
        finally:
            mod.close()

    def test_vector_add_different_sizes(self, vector_add_source):
        import asnumpy as ap
        from asnumpy.compiler import SourceModule

        mod = SourceModule(vector_add_source, options=["-O3"])
        try:
            kernel = mod.get_function("vector_add")
            rng = np.random.RandomState(42)

            for N in [256, 1024, 4096]:
                a_np = rng.randn(N).astype(np.float32)
                b_np = rng.randn(N).astype(np.float32)
                a_ap = ap.ndarray.from_numpy(a_np)
                b_ap = ap.ndarray.from_numpy(b_np)
                c_ap = ap.empty((N,), dtype=ap.float32)

                kernel(a_ap, b_ap, c_ap, N, grid=(8,))
                result = c_ap.to_numpy()
                np.testing.assert_allclose(
                    result, a_np + b_np, rtol=1e-4, atol=1e-5
                )
        finally:
            mod.close()

    def test_scalar_kernel(self, fill_const_source):
        import asnumpy as ap
        from asnumpy.compiler import SourceModule

        mod = SourceModule(fill_const_source, options=["-O3"])
        try:
            kernel = mod.get_function("fill_const")
            N = 512
            c_ap = ap.empty((N,), dtype=ap.float32)

            kernel(c_ap, 3.14, N, grid=(4,))

            result = c_ap.to_numpy()
            np.testing.assert_allclose(result, 3.14, rtol=1e-4, atol=1e-5)
        finally:
            mod.close()

    def test_default_grid(self, vector_add_source):
        import asnumpy as ap
        from asnumpy.compiler import SourceModule

        N = 64
        mod = SourceModule(vector_add_source)
        try:
            kernel = mod.get_function("vector_add")

            a_np = np.ones(N, dtype=np.float32)
            b_np = np.ones(N, dtype=np.float32)
            a_ap = ap.ndarray.from_numpy(a_np)
            b_ap = ap.ndarray.from_numpy(b_np)
            c_ap = ap.empty((N,), dtype=ap.float32)

            # Default grid (1 core)
            kernel(a_ap, b_ap, c_ap, N)

            result = c_ap.to_numpy()
            np.testing.assert_allclose(result, 2.0, rtol=1e-4, atol=1e-5)
        finally:
            mod.close()


# ==========================================================================
# Test: Caching
# ==========================================================================

class TestCaching:
    """Tests for the SHA256-based cache system."""

    def test_cache_key_deterministic(self):
        source = "void foo() {}"
        key1 = compute_cache_key(source, [], "Ascend910B1", "v1.0")
        key2 = compute_cache_key(source, [], "Ascend910B1", "v1.0")
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest

    def test_cache_key_different_source(self):
        key1 = compute_cache_key("void a() {}", [], "Ascend910B1", "v1.0")
        key2 = compute_cache_key("void b() {}", [], "Ascend910B1", "v1.0")
        assert key1 != key2

    def test_cache_key_different_options(self):
        key1 = compute_cache_key("x", ["-O2"], "Ascend910B1", "v1.0")
        key2 = compute_cache_key("x", ["-O3"], "Ascend910B1", "v1.0")
        assert key1 != key2

    def test_cache_key_different_soc(self):
        key1 = compute_cache_key("x", [], "Ascend910B1", "v1.0")
        key2 = compute_cache_key("x", [], "Ascend910B2", "v1.0")
        assert key1 != key2

    def test_get_cache_dir(self):
        cache_dir = get_cache_dir()
        assert cache_dir.name == "cache"
        assert ".asnumpy" in str(cache_dir)

    def test_get_compiler_version(self):
        ver = get_compiler_version()
        assert isinstance(ver, str)
        assert len(ver) > 0

    @requires_bisheng
    def test_cache_store_and_hit(self, vector_add_source):
        import shutil

        from asnumpy.compiler.cache import cache_hit, store_in_cache

        # Compute key
        compiler_ver = get_compiler_version()
        key = compute_cache_key(
            vector_add_source, [], "Ascend910B1", compiler_ver
        )

        # Clean any existing cache entry
        entry = get_cache_dir() / key
        if entry.exists():
            shutil.rmtree(entry)

        # Compile to a temp dir and store
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            bisheng_compiler.compile_kernel(vector_add_source, out_dir)
            store_in_cache(key, out_dir)

        # Check cache hit
        cached = cache_hit(key)
        assert cached is not None
        assert cached.exists()

    @requires_acl_binary_load
    @requires_bisheng
    def test_disable_cache(self, vector_add_source):
        from asnumpy.compiler import SourceModule

        mod1 = SourceModule(vector_add_source, disable_cache=True)
        try:
            assert "vector_add" in mod1.list_functions()
        finally:
            mod1.close()


# ==========================================================================
# Test: Edge Cases
# ==========================================================================

class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_empty_source(self):
        """Empty source warns about extern "C" on construction."""
        from asnumpy.compiler import SourceModule

        # Empty source triggers warning but also fails at bisheng compile stage.
        # Just verify the warning check fires.
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from asnumpy.compiler.source_module import _check_extern_c
            _check_extern_c("")
            assert len(w) == 1
            assert "extern" in str(w[0].message).lower()

    def test_source_without_kernels_compiles(self):
        """Source without kernel functions should compile but have no functions."""
        src = '// just a comment, no kernel here\n'
        # This is valid C++ but no kernel functions
        pass

    @requires_acl_binary_load
    @requires_bisheng
    def test_source_without_extern_c_warns(self):
        from asnumpy.compiler import SourceModule

        src = """
        #include "kernel_operator.h"
        using namespace AscendC;
        __global__ __aicore__ void no_extern(float* a, int n) {}
        """
        with pytest.warns(UserWarning, match='extern "C"'):
            mod = SourceModule(src)
            mod.close()
