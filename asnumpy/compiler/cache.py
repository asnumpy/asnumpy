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

"""SHA256-based caching for compiled Ascend C kernel binaries."""

import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def get_cache_dir() -> Path:
    """Return the AsNumpy kernel cache directory (~/.asnumpy/cache/)."""
    cache = Path.home() / ".asnumpy" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def compute_cache_key(
    source: str,
    options: list[str],
    soc_version: str,
    compiler_version: str,
    include_dirs: list[str] | None = None,
) -> str:
    """Compute a deterministic SHA256 cache key from compilation parameters."""
    data = source + "|" + "|".join(sorted(options or [])) + "|"
    data += soc_version + "|" + compiler_version
    if include_dirs:
        data += "|" + "|".join(sorted(include_dirs))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def get_compiler_version() -> str:
    """Run 'bisheng --version' and return the first line of output."""
    from .bisheng_compiler import _get_bisheng_path

    bisheng = _get_bisheng_path()
    try:
        result = subprocess.run(
            [bisheng, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except Exception:
        return "unknown"


def cache_hit(cache_key: str) -> Path | None:
    """Check if a cached .o file exists for the given key. Returns path or None."""
    entry_dir = get_cache_dir() / cache_key
    o_file = entry_dir / "kernel.so"
    if not o_file.exists():
        return None

    # Wait for any ongoing write to finish (shared lock)
    lock_file = entry_dir / ".lock"
    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    # Re-check after acquiring lock
    return o_file if o_file.exists() else None


def store_in_cache(cache_key: str, build_dir: Path) -> None:
    """Store compiled artifacts from build_dir into the cache.

    Uses an exclusive file lock to prevent concurrent writes to the same entry.
    """
    entry_dir = get_cache_dir() / cache_key
    entry_dir.mkdir(parents=True, exist_ok=True)

    lock_file = entry_dir / ".lock"
    with open(lock_file, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Double-check: another process may have written while we waited
            if (entry_dir / "kernel.o").exists():
                return

            for filename in ["kernel.cpp", "kernel.so", "compile.log"]:
                src = build_dir / filename
                if src.exists():
                    shutil.copy2(src, entry_dir / filename)

            # Write metadata
            meta = {
                "cached_at": str(build_dir),
            }
            (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
