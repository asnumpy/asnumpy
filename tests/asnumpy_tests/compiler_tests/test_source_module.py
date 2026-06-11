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

"""Tests for the asnumpy.compiler module (simplified API).

Test classes organized by dependency:
- L0: pure-Python helpers (kernel name discovery, etc.)
- L1: bisheng compilation
- L2: SourceModule integration (mock RTS)
- L3: KernelFunction argument marshalling (mock RTS)
"""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asnumpy.compiler import KernelFunction, SourceModule
from asnumpy.compiler import source_module as sm

# ==========================================================================
# Skip markers
# ==========================================================================


def _bisheng_available() -> bool:
    try:
        sm._find_bisheng()
        return True
    except Exception:
        return False


requires_bisheng = pytest.mark.skipif(
    not _bisheng_available(),
    reason="Requires bisheng compiler",
)

requires_kernel_exec = pytest.mark.skip(
    reason="Bisheng compiler on 910B4 generates illegal AI Core instructions "
    "for arithmetic ops. TODO: retest with fixed bisheng compiler."
)


# ==========================================================================
# L0: Pure-Python helpers
# ==========================================================================

class TestKernelDiscovery:
    """Tests for _find_kernels (no compilation needed)."""

    def test_find_single_kernel(self, vector_add_source):
        kernels = sm._find_kernels(vector_add_source)
        assert kernels == ["vector_add"]

    def test_find_multi_kernel(self, multi_kernel_source):
        kernels = sm._find_kernels(multi_kernel_source)
        assert "kernel_one" in kernels
        assert "kernel_two" in kernels
        assert len(kernels) == 2

    def test_find_no_kernel(self):
        kernels = sm._find_kernels("// just a comment")
        assert kernels == []

    def test_find_no_extern_c(self):
        src = '__global__ __aicore__ void no_extern(__gm__ float* a, int n) {}'
        kernels = sm._find_kernels(src)
        assert kernels == []  # regex requires 'extern "C"' prefix


# ==========================================================================
# L1: Bisheng compilation
# ==========================================================================

class TestCompilation:
    """Tests for _compile + _extract_elf (needs bisheng compiler)."""

    @requires_bisheng
    def test_compiles_without_error(self, vector_add_source):
        o_file = sm._compile(vector_add_source, None, "Ascend910B4", "VecCore")
        assert o_file.exists()
        assert o_file.suffix == ".o"

    @requires_bisheng
    def test_compiled_o_is_elf(self, vector_add_source):
        o_file = sm._compile(vector_add_source, None, "Ascend910B4", "VecCore")
        header = o_file.read_bytes()[:4]
        assert header == b"\x7fELF"

    @requires_bisheng
    def test_compile_invalid_source_raises(self):
        with pytest.raises(RuntimeError, match="bisheng compilation failed"):
            sm._compile("this is not valid C++ code", None, "Ascend910B4", "VecCore")

    @requires_bisheng
    def test_extract_elf_produces_bytes(self, vector_add_source):
        o_file = sm._compile(vector_add_source, None, "Ascend910B4", "VecCore")
        elf_data = sm._extract_elf(o_file)
        assert isinstance(elf_data, bytes)
        assert len(elf_data) > 0
        # Check ELF magic
        assert elf_data[:4] == b"\x7fELF"

    @requires_bisheng
    def test_compile_with_options(self, vector_add_source):
        o_file = sm._compile(vector_add_source, ["-O3"], "Ascend910B4", "VecCore")
        assert o_file.exists()


# ==========================================================================
# L2: SourceModule integration (mock RTS — no NPU needed)
# ==========================================================================

class TestSourceModuleMock:
    """Tests SourceModule using mocked _rt module."""

    @pytest.fixture(autouse=True)
    def _patch_rt(self):
        """Patch _rt for all tests in this class."""
        mock = MagicMock(name="_rt")
        mock.register_binary.return_value = 42
        mock.register_function.return_value = None
        mock.unregister_binary.return_value = None
        mock.launch_kernel.return_value = None
        with patch("asnumpy.compiler.source_module._rt", mock):
            self._mock_rt = mock
            yield
        # Cleanup after test
        self._mock_rt = None

    @requires_bisheng
    def test_get_function_returns_kernel(self, vector_add_source):
        mod = SourceModule(vector_add_source)
        kernel = mod.get_function("vector_add")
        assert isinstance(kernel, KernelFunction)
        assert kernel.name == "vector_add"

    @requires_bisheng
    def test_get_function_with_signature(self, vector_add_source):
        mod = SourceModule(vector_add_source)
        sig = ["float32*", "float32*", "float32*", "int32"]
        kernel = mod.get_function("vector_add", signature=sig)
        assert kernel._signature == sig

    @requires_bisheng
    def test_get_function_unknown_raises(self, vector_add_source):
        mod = SourceModule(vector_add_source)
        with pytest.raises(ValueError, match="not found"):
            mod.get_function("nonexistent")

    @requires_bisheng
    def test_multi_kernel_source(self, multi_kernel_source):
        mod = SourceModule(multi_kernel_source)
        k1 = mod.get_function("kernel_one")
        k2 = mod.get_function("kernel_two")
        assert k1.name == "kernel_one"
        assert k2.name == "kernel_two"
        # register_function should be called for each kernel
        assert self._mock_rt.register_function.call_count >= 2

    @requires_bisheng
    def test_register_binary_called(self, vector_add_source):
        mod = SourceModule(vector_add_source)
        self._mock_rt.register_binary.assert_called_once()
        self._mock_rt.register_function.assert_called_once_with(42, "vector_add")

    @requires_bisheng
    def test_extern_c_warning(self):
        # Valid Ascend C source without extern "C" triggers warning
        src_bad = """#include "kernel_operator.h"
        using namespace AscendC;
        __global__ __aicore__ void k(__gm__ float* a, int n) {}
        """

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SourceModule(src_bad)
            warn_msgs = [str(x.message) for x in w if "extern" in str(x.message).lower()]
            assert len(warn_msgs) >= 1

        # Source with extern "C" should NOT warn
        src_ok = """#include "kernel_operator.h"
        using namespace AscendC;
        extern "C" __global__ __aicore__ void k(__gm__ float* a, int n) {}
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SourceModule(src_ok)
            warn_msgs = [str(x.message) for x in w if "extern" in str(x.message).lower()]
            assert len(warn_msgs) == 0


# ==========================================================================
# L3: KernelFunction argument marshalling
# ==========================================================================

class TestKernelFunctionMarshal:
    """Tests KernelFunction._pack_args (no compilation, no NPU)."""

    class FakeNDArray:
        def __init__(self, ptr=0xDEAD0000):
            self.device_address = ptr

    def test_pointer_args_pack_device_address(self):
        sig = ["float32*", "float32*", "float32*", "int32"]
        kernel = KernelFunction("test", sig)
        packed = kernel._pack_args((
            self.FakeNDArray(0x1000),
            self.FakeNDArray(0x2000),
            self.FakeNDArray(0x3000),
            1024,
        ))
        assert len(packed) == 4
        # Pointer args: 8 bytes each (little-endian uint64)
        assert len(packed[0]) == 8
        assert len(packed[1]) == 8
        assert len(packed[2]) == 8
        # Scalar (int32): 4 bytes
        assert len(packed[3]) == 4
        # Verify pointer values
        assert struct.unpack("<Q", packed[0])[0] == 0x1000
        assert struct.unpack("<Q", packed[1])[0] == 0x2000
        assert struct.unpack("<Q", packed[2])[0] == 0x3000

    def test_pointer_arg_raises_for_non_ndarray(self):
        sig = ["float32*", "float32*"]
        kernel = KernelFunction("test", sig)
        with pytest.raises(TypeError, match="ndarray"):
            kernel._pack_args(([1, 2, 3], None))

    def test_wrong_arg_count_raises(self):
        sig = ["float32*", "float32*", "int32"]
        kernel = KernelFunction("test", sig)
        with pytest.raises(TypeError, match="expects 3"):
            kernel._pack_args((1, 2))

    def test_scalar_types(self):
        sig = ["int64", "float64", "bool"]
        kernel = KernelFunction("test", sig)
        packed = kernel._pack_args((np.int64(42), np.float64(3.14), True))
        assert len(packed[0]) == 8  # int64
        assert len(packed[1]) == 8  # float64
        assert len(packed[2]) == 1  # bool

    def test_no_signature_defaults_to_int32(self):
        """Without a signature, scalars default to int32 (4 bytes)."""
        kernel = KernelFunction("test")
        packed = kernel._pack_args((42,))
        assert len(packed[0]) == 4  # defaults to int32

    def test_mixed_args(self):
        sig = ["float32*", "float32", "int32"]
        kernel = KernelFunction("test", sig)
        packed = kernel._pack_args((self.FakeNDArray(0xBEEF), 3.14, 512))
        assert len(packed[0]) == 8  # pointer
        assert len(packed[1]) == 4  # float32
        assert len(packed[2]) == 4  # int32


# ==========================================================================
# L4: KernelFunction call (mock RTS launch)
# ==========================================================================

class TestKernelFunctionCall:
    """Tests KernelFunction.__call__ using mocked _rt.launch_kernel."""

    class FakeNDArray:
        def __init__(self, ptr=0xDEAD0000):
            self.device_address = ptr

    @pytest.fixture(autouse=True)
    def _patch_rt(self):
        mock = MagicMock(name="launch_kernel")
        mock.return_value = None
        with patch("asnumpy.compiler.kernel_function._rt.launch_kernel", mock):
            self._mock_launch = mock
            yield

    def test_call_launches_with_grid(self):
        sig = ["float32*", "float32*", "float32*", "int32"]
        kernel = KernelFunction("vector_add", sig)
        kernel(
            self.FakeNDArray(), self.FakeNDArray(), self.FakeNDArray(), 1024,
            grid=(8,),
        )
        self._mock_launch.assert_called_once()
        call_args = self._mock_launch.call_args
        assert call_args[0][0] == "vector_add"  # kernel name
        assert call_args[0][1] == 8             # block_dim from grid

    def test_call_default_grid_is_one(self):
        sig = ["float32*", "int32"]
        kernel = KernelFunction("test", sig)
        kernel(self.FakeNDArray(), 64)
        call_args = self._mock_launch.call_args
        assert call_args[0][1] == 1  # default block_dim

    def test_call_passes_packed_args(self):
        sig = ["float32*", "int32"]
        kernel = KernelFunction("test", sig)
        kernel(self.FakeNDArray(0xABCD), 42, grid=(4,))
        call_args = self._mock_launch.call_args
        packed = call_args[0][2]  # packed_args
        assert len(packed) == 2
        assert struct.unpack("<Q", packed[0])[0] == 0xABCD  # pointer
        assert len(packed[1]) == 4  # int32 scalar

    def test_repr(self):
        kernel = KernelFunction("my_kernel", ["float32*", "int32"])
        repr_str = repr(kernel)
        assert "my_kernel" in repr_str
        assert "float32*" in repr_str
        assert "int32" in repr_str
