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
            o_file = bisheng_compiler.compile_kernel(
                vector_add_source, out_dir
            )
            assert o_file.exists()
            assert o_file.suffix == ".o"

    @requires_bisheng
    def test_compiled_o_is_elf(self, vector_add_source):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            o_file = bisheng_compiler.compile_kernel(
                vector_add_source, out_dir
            )
            # Check ELF magic bytes
            header = o_file.read_bytes()[:4]
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

    @requires_bisheng
    def test_disable_cache(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            try:
                assert "vector_add" in mod.list_functions()
            finally:
                mod.close()


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

    @requires_bisheng
    def test_source_without_extern_c_warns(self, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        src = """
        #include "kernel_operator.h"
        using namespace AscendC;
        __global__ __aicore__ void no_extern(__gm__ float* a, int n) {}
        """
        with _mock_get_lib(sm, mock_compiler_lib):
            with pytest.warns(UserWarning, match='extern "C"'):
                mod = SourceModule(src, disable_cache=True)
                mod.close()


# ==========================================================================
# Test: SourceModule with mock C extension
# ==========================================================================


class TestSourceModuleMock:
    """Tests SourceModule orchestration using a mock C extension.

    These replace the ``TestSourceModule`` tests (which require CANN 8.5+)
    by mocking ``_get_lib()``.  Real bisheng compilation is still used so
    we validate against a genuine .o file and its symbol table.
    """

    @requires_bisheng
    def test_compilation_and_loading(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, keep=True, disable_cache=True)
            try:
                assert mod.list_functions() is not None
                assert "vector_add" in mod.list_functions()
                # Verify C extension calls
                mock_compiler_lib.load_binary.assert_called_once()
                mock_compiler_lib.get_function.assert_called()
            finally:
                mod.close()

    @requires_bisheng
    def test_get_function_returns_callable(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            try:
                kernel = mod.get_function("vector_add")
                assert isinstance(kernel, KernelFunction)
                assert kernel.name == "vector_add"
            finally:
                mod.close()

    @requires_bisheng
    def test_get_function_unknown_name_raises(
        self, vector_add_source, mock_compiler_lib
    ):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            try:
                with pytest.raises(ValueError, match="not found"):
                    mod.get_function("nonexistent")
            finally:
                mod.close()

    @requires_bisheng
    def test_close_calls_unload(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            bin_handle = mod._bin_handle
            mod.close()
            mock_compiler_lib.unload_binary.assert_called_with(bin_handle)
            # Safe to call again — should not call unload twice
            mock_compiler_lib.unload_binary.reset_mock()
            mod.close()
            mock_compiler_lib.unload_binary.assert_not_called()

    @requires_bisheng
    def test_context_manager(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            with SourceModule(vector_add_source, disable_cache=True) as mod:
                assert len(mod.list_functions()) > 0
            # Exiting context manager calls close, which calls unload
            mock_compiler_lib.unload_binary.assert_called_once()

    @requires_bisheng
    def test_multi_kernel_source(self, multi_kernel_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(multi_kernel_source, disable_cache=True)
            try:
                funcs = mod.list_functions()
                assert "kernel_one" in funcs
                assert "kernel_two" in funcs
                # get_function should be called for each discovered kernel
                assert mock_compiler_lib.get_function.call_count >= 2
            finally:
                mod.close()

    @requires_bisheng
    def test_manual_signature(self, vector_add_source, mock_compiler_lib):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            try:
                sig = ["float32*", "float32*", "float32*", "int32"]
                kernel = mod.get_function("vector_add", signature=sig)
                assert len(kernel._arg_specs) == 4
                # Verify manual signature entries
                assert kernel._arg_specs[0].arg_type == "float32*"
                assert kernel._arg_specs[3].arg_type == "int32"
            finally:
                mod.close()

    @requires_bisheng
    def test_list_functions_empty_for_no_kernel_source(self, mock_compiler_lib):
        """Source with no kernel functions should produce an empty list."""
        from asnumpy.compiler import SourceModule, source_module as sm

        src = '// just a comment, no kernel here\n'
        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(src, disable_cache=True)
            try:
                assert mod.list_functions() == []
            finally:
                mod.close()

    @requires_bisheng
    def test_disable_cache_skips_cache(
        self, vector_add_source, mock_compiler_lib
    ):
        from asnumpy.compiler import SourceModule, source_module as sm

        with _mock_get_lib(sm, mock_compiler_lib):
            mod = SourceModule(vector_add_source, disable_cache=True)
            try:
                assert "vector_add" in mod.list_functions()
                # With disable_cache=True, load_binary still called
                mock_compiler_lib.load_binary.assert_called_once()
            finally:
                mod.close()


# ==========================================================================
# Test: KernelFunction / PreparedKernel with mock C extension
# ==========================================================================


class TestKernelFunctionMock:
    """Tests KernelFunction argument marshalling and launch using a mock C extension.

    These replace ``TestKernelExecution`` (requires NPU) by verifying that
    the correct arguments are packed and passed to ``launch_kernel``.
    """

    @pytest.fixture
    def func_handle(self):
        return 42

    @pytest.fixture
    def float_ptr_specs(self):
        return [
            ArgSpec("a", "float*", True, 8),
            ArgSpec("b", "float*", True, 8),
            ArgSpec("c", "float*", True, 8),
            ArgSpec("n", "int32", False, 4),
        ]

    @pytest.fixture
    def mixed_specs(self):
        return [
            ArgSpec("out", "float*", True, 8),
            ArgSpec("value", "float32", False, 4),
            ArgSpec("n", "int32", False, 4),
        ]

    def test_marshal_pointer_args_packs_device_address(
        self, func_handle, float_ptr_specs
    ):
        """Pointer arguments should be packed as 8-byte device addresses."""

        class FakeNDArray:
            device_address = 0xABCD00001234

        kernel = KernelFunction("test", func_handle, float_ptr_specs)
        packed = kernel._marshal_args((
            FakeNDArray(), FakeNDArray(), FakeNDArray(), 1024
        ))

        assert len(packed) == 4
        # Pointer args: 8 bytes each
        for i in range(3):
            assert len(packed[i]) == 8
        # Scalar arg (int32): 4 bytes
        assert len(packed[3]) == 4

    def test_marshal_pointer_arg_raises_for_non_ndarray(
        self, func_handle, float_ptr_specs
    ):
        """Passing a non-ndarray for a pointer arg should raise TypeError."""
        kernel = KernelFunction("test", func_handle, float_ptr_specs)
        with pytest.raises(TypeError, match="ndarray"):
            kernel._marshal_args(([1, 2, 3], None, None, 1))

    def test_marshal_wrong_arg_count_raises(self, func_handle, float_ptr_specs):
        """Mismatched arg count should raise TypeError."""
        kernel = KernelFunction("test", func_handle, float_ptr_specs)
        with pytest.raises(TypeError, match="expects 4"):
            kernel._marshal_args((1, 2, 3))

    def test_marshal_mixed_args(self, func_handle, mixed_specs):
        """Mixed pointer + scalar kernel: fill_const(out, value, n)."""

        class FakeNDArray:
            device_address = 0xBEEF

        kernel = KernelFunction("fill_const", func_handle, mixed_specs)
        packed = kernel._marshal_args((FakeNDArray(), 3.14, 512))

        # Pointer: 8 bytes
        assert len(packed[0]) == 8
        # float32 scalar: 4 bytes
        assert len(packed[1]) == 4
        # int32 scalar: 4 bytes
        assert len(packed[2]) == 4

    def test_call_launches_kernel(self, func_handle, float_ptr_specs,
                                   mock_compiler_lib):
        """__call__ should invoke launch_kernel via mocked C extension."""
        from asnumpy.compiler import kernel_function as kf

        class FakeNDArray:
            device_address = 0xDEAD0000

        kernel = KernelFunction("test", func_handle, float_ptr_specs)

        with _mock_get_lib(kf, mock_compiler_lib):
            kernel(
                FakeNDArray(), FakeNDArray(), FakeNDArray(), 1024,
                grid=(8,),
            )

        mock_compiler_lib.launch_kernel.assert_called_once()
        call_args = mock_compiler_lib.launch_kernel.call_args
        # First positional arg is func_handle
        assert call_args[0][0] == func_handle
        # Second is block_dim (from grid)
        assert call_args[0][1] == 8

    def test_call_default_grid_is_one(self, func_handle, float_ptr_specs,
                                       mock_compiler_lib):
        """If no grid is passed, block_dim should default to 1."""
        from asnumpy.compiler import kernel_function as kf

        class FakeNDArray:
            device_address = 0x1000

        kernel = KernelFunction("test", func_handle, float_ptr_specs)

        with _mock_get_lib(kf, mock_compiler_lib):
            kernel(FakeNDArray(), FakeNDArray(), FakeNDArray(), 64)

        call_args = mock_compiler_lib.launch_kernel.call_args
        assert call_args[0][1] == 1  # block_dim == 1

    def test_marshal_scalar_types(self, func_handle):
        """Verify that int64, float64, bool scalars are packed correctly."""
        specs = [
            ArgSpec("a", "int64", False, 8),
            ArgSpec("b", "float64", False, 8),
            ArgSpec("c", "bool", False, 1),
        ]
        kernel = KernelFunction("test", func_handle, specs)
        packed = kernel._marshal_args((np.int64(42), np.float64(3.14), True))

        assert len(packed[0]) == 8  # int64
        assert len(packed[1]) == 8  # float64
        assert len(packed[2]) == 1  # bool


class TestPreparedKernelMock:
    """Tests PreparedKernel event management and timing using a mock C extension."""

    @pytest.fixture
    def kernel(self):
        specs = [ArgSpec("a", "int32", False, 4)]
        return KernelFunction("dummy", 99, specs)

    def test_prepare_creates_events_and_stream(self, kernel, mock_compiler_lib):
        from asnumpy.compiler import kernel_function as kf

        with _mock_get_lib(kf, mock_compiler_lib):
            pk = kernel.prepare()

        mock_compiler_lib.create_event.assert_called()
        mock_compiler_lib.create_stream.assert_called_once()
        # Two events: start + end
        assert mock_compiler_lib.create_event.call_count == 2
        # Prevent __del__ from calling real C extension after mock exits
        _null_prepared_kernel_handles(pk)

    def test_call_records_events_and_measures_elapsed(
        self, kernel, mock_compiler_lib
    ):
        from asnumpy.compiler import kernel_function as kf

        with _mock_get_lib(kf, mock_compiler_lib):
            pk = kernel.prepare()
            pk(42)

        # Should record start + end
        assert mock_compiler_lib.record_event.call_count == 2
        # Should synchronize end event
        mock_compiler_lib.synchronize_event.assert_called_once()
        # Should measure elapsed time
        mock_compiler_lib.elapsed_time_between.assert_called_once()
        _null_prepared_kernel_handles(pk)

    def test_time_property_returns_elapsed(self, kernel, mock_compiler_lib):
        from asnumpy.compiler import kernel_function as kf

        mock_compiler_lib.elapsed_time_between.return_value = 3.75

        with _mock_get_lib(kf, mock_compiler_lib):
            pk = kernel.prepare()
            pk(42)

        assert pk.time == 3.75
        _null_prepared_kernel_handles(pk)

    def test_del_cleans_up_events_and_stream(self, kernel, mock_compiler_lib):
        from asnumpy.compiler import kernel_function as kf

        with _mock_get_lib(kf, mock_compiler_lib):
            pk = kernel.prepare()
            start_handle = pk._start_event
            end_handle = pk._end_event
            stream_handle = pk._stream
            pk.__del__()
            # Null handles so gc __del__ is a no-op
            pk._start_event = None
            pk._end_event = None
            pk._stream = None

        mock_compiler_lib.destroy_event.assert_any_call(start_handle)
        mock_compiler_lib.destroy_event.assert_any_call(end_handle)
        mock_compiler_lib.destroy_stream.assert_called_with(stream_handle)


# ==========================================================================
# Mock helper
# ==========================================================================


from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _mock_get_lib(module, mock_lib):
    """Temporarily replace ``module._get_lib`` with a lambda returning ``mock_lib``."""
    with patch.object(module, "_get_lib", return_value=mock_lib):
        yield


def _null_prepared_kernel_handles(pk):
    """Prevent ``PreparedKernel.__del__`` from calling real C extension.

    After a mock-based test, ``__del__`` would call the real ``destroy_event``
    / ``destroy_stream`` with fake handles, causing a segfault.  Nulling the
    handles makes ``__del__`` a no-op.
    """
    pk._start_event = None
    pk._end_event = None
    pk._stream = None
