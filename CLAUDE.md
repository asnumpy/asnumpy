# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AsNumpy is a NumPy-compatible library for Huawei Ascend NPU (910B) acceleration. It provides a three-layer architecture:
- **Python frontend** (`src/asnumpy/`) - NumPy-compatible API
- **Pybind11 bindings** (`bindings/python/`) - C++ to Python interface
- **C++ core** (`csrc/`, `include/`) - CANN ACLNN operator wrappers

The core data structure is `NPUArray`, which mirrors `numpy.ndarray` but executes operations on the NPU via CANN runtime.

## Build System

Uses **scikit-build-core** (CMake + Python build frontend), not traditional setuptools.

### Environment Setup

Set `ASCEND_TOOLKIT_HOME` (or `ASCEND_HOME_PATH`) before building when the CANN SDK is not in the default location:
```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
```
If neither variable is set, CMake falls back to `/usr/local/Ascend/ascend-toolkit/latest` with a warning.

### Build Commands

**Development build (editable install)**
```bash
pip install -e .
```

**Full rebuild**
```bash
pip uninstall asnumpy -y && pip install -e .
```

**Production wheel build**
```bash
python -m build
pip install dist/*.whl
```

The build system uses scikit-build-core's editable mode with "redirect" strategy, so code changes in `src/asnumpy/*.py` are reflected immediately without rebuild. C++ changes require `pip install -e .` again.

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Module
```bash
# Math module tests
pytest tests/asnumpy_tests/math_tests/

# Logic module tests
pytest tests/asnumpy_tests/logic_tests/
```

### Run Single Test File or Function
```bash
# Single file
pytest tests/asnumpy_tests/math_tests/test_other_special_functions.py

# Single function
pytest tests/asnumpy_tests/math_tests/test_other_special_functions.py::test_sinc_basic
```

### Parallel Testing
```bash
# Use pytest-xdist for parallel execution
pytest tests/ -n auto
```

### Test Options
```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Multi-NPU tests (requires multiple NPU devices)
pytest --multi-npu

# Specify NPU device
pytest --npu-id 1
```

## Code Quality

### Linting and Formatting

**Run all pre-commit hooks manually**
```bash
pre-commit run --all-files
```

**Individual tools**
```bash
# Python linting (ruff)
ruff check src/ tests/

# Python formatting (ruff)
ruff format src/ tests/

# Type checking (mypy)
mypy src/

# C++ formatting (clang-format)
clang-format -i csrc/**/*.cpp include/**/*.hpp
```

Pre-commit hooks run automatically on `git commit` if installed via `pre-commit install`.

## Architecture

### Three-Layer Design

1. **Python Frontend** (`src/asnumpy/`)
   - `__init__.py` - Public API exports via lazy loading
   - `math.py` - Mathematical functions (sin, cos, exp, log, etc.)
   - `linalg/` - Linear algebra
   - `logic.py` - Logical operations
   - `array.py` - Array creation functions
   - `random/` - Random sampling
   - `statistics.py` - Statistical functions
   - `nn.py` - Neural network ops (softmax)
   - `utils.py` - Core `ndarray` class wrapper

2. **Pybind11 Bindings** (`bindings/python/`)
   - `module.cpp` - Main module definition
   - `bind_math.cpp` - Math function bindings
   - `bind_logic.cpp` - Logic function bindings
   - `bind_array.cpp` - Array creation bindings
   - `bind_utils.cpp` - NPUArray class binding
   - Each `bind_*.cpp` exposes C++ functions to Python

3. **C++ Core** (`csrc/`, `include/`)
   - `utils/npu_array.cpp` - Core `NPUArray` class (RAII, device memory, CANN tensor management)
   - `cann/driver.cpp` - Device initialization and lifecycle
   - `math/` - Math operator implementations
   - `logic/` - Logic operator implementations
   - `dtypes/` - Data type system and promotion rules

### NPUArray Design

`NPUArray` is the core data structure with:
- **dtype** (py::dtype) - Element type
- **shape** (std::vector<int64_t>) - Dimensions
- **strides** (std::vector<int64_t>) - Memory layout, **in elements, not bytes**. Byte strides exist only at the Python boundary.
- **tensorPtr** (aclTensor*) - CANN tensor handle, pointing at `buffer + byteOffset`
- **buffer** (std::shared_ptr\<DeviceBuffer\>, private) - Refcounted device allocation
- **byteOffset** (int64_t, private) - Start of this array's data within `buffer`

**Shared storage, zero-copy views.** `Reshape`, `Flatten`, `Ravel` and supported
`GetItem` operations return views: they share `buffer` with their source and allocate
nothing. A view and its source alias the same memory. AsNumpy deliberately differs
from NumPy for `Flatten`: NumPy's `flatten()` copies, while AsNumpy's is zero-copy.
Nothing can currently write in place (there is no `__setitem__` and `multiply()`
rejects `out=`), but any future in-place API must honor the shared-storage rule.
`Transpose` is the deliberate exception — it materializes, because a permuted array
is not C-contiguous and downstream operators assume they are.

**Memory pooling.** NPUArray device buffers and standard `AclWorkspace` allocations go
through `asnumpy::DeviceMemoryPool` (`include/asnumpy/utils/device_memory_pool.hpp`),
not directly through `aclrtMalloc`. The pool caches freed blocks by exact byte size
(capped at 256 MB, evicting largest-first), because the driver's allocation cost is not
proportional to the request — 64 MB measured 399 µs against 15 µs for 4 MB, and every
operator allocates at least one output. If you add a place that needs device memory,
go through the pool and call `Clear()` before any device reset or finalization.
Specialized operator bodies may still manage workspaces directly with their own
`aclrtMalloc`/`aclrtFree` pairs.

**RAII resource management**: the destructor calls `aclDestroyTensor`; `DeviceBuffer`'s
destructor returns the allocation to the pool when the last referencing array dies. No
manual cleanup needed, and a view stays valid after the array it was made from is gone.

**Data transfer**:
- `FromNumpy(np.ndarray)` → NPU (ACL_MEMCPY_HOST_TO_DEVICE)
- `ToNumpy()` → CPU (ACL_MEMCPY_DEVICE_TO_HOST)

### Directory Structure

```
csrc/
├── array/          # Array creation ops
├── cann/           # CANN driver and device management
├── dtypes/         # Data type system
├── linalg/         # Linear algebra ops
├── logic/          # Logical ops
├── math/           # Math ops (largest module)
├── nn/             # Neural network ops
├── random/         # Random sampling
├── sorting/        # Sorting ops
├── statistics/     # Statistical ops
└── utils/          # NPUArray, ACL resources, error handling

include/asnumpy/
├── array/
├── cann/
├── dtypes/
├── linalg/
├── logic/
├── math/
├── nn/
├── random/
├── sorting/
├── statistics/
└── utils/

bindings/python/
├── module.cpp      # PYBIND11_MODULE(_core, ...)
├── bind_*.cpp      # Per-module bindings

src/asnumpy/
├── __init__.py     # Main exports
├── math.py         # Math function wrappers
├── logic.py        # Logic function wrappers
├── array.py        # Array creation wrappers
├── utils.py        # ndarray class
└── testing/        # Test utilities and decorators
```

## Development Workflow

### Adding a New Operator

Follow this sequence (example: adding `sinc` function):

1. **Add C++ declaration** (`include/asnumpy/math/other_special_functions.hpp`)
   ```cpp
   NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);
   ```

2. **Implement C++ function** (`csrc/math/other_special_functions.cpp`)
   - Determine output dtype
   - Create output NPUArray
   - Call CANN operator (e.g., `aclnnSinc`)
   - Handle workspace allocation
   - Synchronize device
   - Return result

3. **Add pybind11 binding** (`bindings/python/bind_math.cpp`)
   ```cpp
   math.def("sinc", &Sinc, py::arg("x"), py::arg("dtype") = py::none());
   ```

4. **Add Python wrapper** (`src/asnumpy/math.py`)
   ```python
   from ._core.math import sinc as _sinc

   def sinc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
       return ndarray(_sinc(x, _convert_dtype(dtype)))
   ```

5. **Register the public export** (`src/asnumpy/__init__.py`)
   Add the function to `_LAZY_MAPPING` and optionally to the `TYPE_CHECKING` imports for static analysis:
   ```python
   _LAZY_MAPPING = {
       # ...
       "sinc": ".math",
   }
   ```
   `__all__` is generated from `_LAZY_MAPPING`, and `__getattr__` imports the function on first access.

6. **Write tests** (`tests/asnumpy_tests/math_tests/test_other_special_functions.py`)
   ```python
   @testing.for_float_dtypes(no_float16=True)
   @testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
   def test_sinc_basic(xp, dtype):
       # Test implementation
   ```

7. **Rebuild and test**
   ```bash
   pip install -e .
   pytest tests/asnumpy_tests/math_tests/test_other_special_functions.py::test_sinc_basic
   ```

### CANN Operator Pattern

For common unary and binary operators, prefer the `EXECUTE_UNARY_OP` and
`EXECUTE_BINARY_OP` helpers in `include/asnumpy/utils/acl_executor.hpp`, or a
dtype-promoting wrapper such as `UnaryFloatingPromoteOp`. They handle output creation,
workspace lifetime, error checks, and device synchronization. A manual ACLNN sequence
should use the same RAII and checked-error pattern:

```cpp
// 1. Determine dtypes and create the output array
NPUArray out(x.shape, out_py_dtype);

// 2. Get workspace size and executor
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
auto error = aclnnOperatorGetWorkspaceSize(
    x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
);
ACLNN_CHECK(error, "aclnnOperatorGetWorkspaceSize");

// 3. Allocate pooled workspace with RAII
asnumpy::AclWorkspace workspace(workspaceSize);

// 4. Execute operator
error = aclnnOperator(workspace.get(), workspaceSize, executor, nullptr);
ACLNN_CHECK(error, "aclnnOperator");

// 5. Synchronize device; workspace is released automatically on return or throw
error = aclrtSynchronizeDevice();
ACL_RT_CHECK(error, "aclnnOperator: aclrtSynchronizeDevice");

return out;
```

**Error handling**: Check ACL and ACLNN return values with `ACL_RT_CHECK` and
`ACLNN_CHECK`; these include source context and details from `aclGetRecentErrMsg()`.

### Testing Utilities

AsNumpy provides decorators for parameterized testing:

**Data type decorators**:
- `@testing.for_float_dtypes(no_float16=True)` - float32, float64
- `@testing.for_int_dtypes()` - int8, int16, int32, int64, uint8, uint16, uint32, uint64
- `@testing.for_all_dtypes()` - Broad default set; excludes float16, uint32, and uint64 unless their `no_*` options are overridden, and includes bool and complex by default
- `@testing.for_dtypes([np.float32, np.int32])` - Specific types

**Comparison decorators**:
- `@testing.numpy_asnumpy_allclose(rtol=1e-7, atol=0)` - Float comparison with tolerance
- `@testing.numpy_asnumpy_array_equal()` - Exact comparison

**Test pattern**:
```python
import numpy as np

@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_operator(xp, dtype):
    """The comparison decorator runs the body with NumPy and AsNumpy."""
    a_np = np.array([1.0, 2.0, 3.0], dtype=dtype)
    a = a_np if xp is np else xp.ndarray.from_numpy(a_np)
    return xp.sinc(a)  # Automatically compares NumPy vs AsNumpy results
```

## Branch Management

- **Main branch**: `master` (all PRs target this)
- **Feature branches**: `feature/<name>`
- **Bug fixes**: `fix/<name>`
- **Release branches**: `release/vX.Y.Z` (for stabilization)
- **Hotfix branches**: `hotfix/<name>` (backport fixes to master)

**Use squash merges for all pull requests.** The repository allows only squash merging. Each PR is combined into one commit on `master`, so new changes form a linear history; existing merge commits remain unchanged. Do not create merge commits or use rebase merging.

**Commit hygiene**: Keep commits within a PR logical and easy to review, but there is no need to squash them locally before merging. Write a clear PR title and description, since they are used for the squashed commit message.

## Important Constraints

1. **Python >= 3.10** - Uses modern type hints and features
2. **CANN >= 8.2.RC1.alpha003** - Required for operator support
3. **GCC >= 11.2** - C++17 features
4. **CMake >= 3.22** - scikit-build-core requirement
5. **Ascend 910B NPU** - Hardware target

## Symbol Visibility

The build uses `-fvisibility=hidden` and `-fvisibility-inlines-hidden` globally. This prevents symbol conflicts with other extensions (especially fmt/spdlog). Only `PyInit__core` is exported. Never change `CMAKE_CXX_VISIBILITY_PRESET` - see CMakeLists.txt comments for rationale.

## Dependencies

**Build-time**:
- scikit-build-core >= 1.0.0
- pybind11 >= 3.0.0
- numpy >= 2.0
- fmt (system package)
- spdlog (system package, with SPDLOG_FMT_EXTERNAL)

**Runtime**:
- numpy >= 2.0
- loguru >= 0.7.0
- CANN runtime libraries

## OpenBOAT Integration

When CANN built-in operators don't cover a NumPy API, AsNumpy uses [OpenBOAT](https://gitcode.com/HIT1920/OpenBOAT) - an open-source Ascend C operator library developed by the same research group (AISS Group, HIT). OpenBOAT operators are compiled and linked separately but follow the same ACLNN interface pattern.

## Versioning

Version is managed by `setuptools-scm` from git tags:
- Tag format: `vX.Y.Z` (e.g., `v0.2.0`)
- Version written to `src/asnumpy/_version.py` at build time
- An exact release tag produces version `X.Y.Z`; untagged development builds produce PEP 440 pre-release versions without the `+g<commit>` local segment
- No manual version editing needed

## Common Issues

**Build fails because the CANN SDK is not found**: Set `ASCEND_TOOLKIT_HOME` or `ASCEND_HOME_PATH` to the CANN toolkit location, or install it at the default path shown above.

**Import error after C++ changes**: Rebuild with `pip install -e .` - Python changes are live, C++ changes need recompilation.

**Test fails with device error**: Check NPU device availability with `npu-smi info`, ensure CANN runtime is properly configured.

**Float16 precision issues**: Many operators have reduced precision or unsupported operations for float16. Use `no_float16=True` in test decorators when appropriate.
