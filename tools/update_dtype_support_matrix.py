from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone


DTYPES: list[tuple[str, str, int]] = [
    # name, kind, acl_enum
    ("bfloat16", "f", 27),
    ("float8_e5m2", "f", 35),
    ("float8_e4m3fn", "f", 36),
    ("float8_e8m0", "f", 37),
    ("float6_e3m2fn", "f", 38),
    ("float6_e2m3fn", "f", 39),
    ("float4_e2m1fn", "f", 40),
    ("float4_e1m2fn", "f", 41),
    ("int4", "i", 29),
    ("uint1", "u", 30),
]


API_ORDER = [
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "eye",
    "identity",
    "linspace",
]


API_NOTES: dict[str, list[str]] = {
    "empty": [
        "Allocation + dtype mapping only; does not require a fill kernel.",
    ],
    "empty_like": [
        "Allocation + dtype mapping only; does not require a fill kernel.",
    ],
    "zeros": [
        "Depends on backend zero-fill kernel support (e.g., aclnnInplaceZero*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "zeros_like": [
        "Depends on backend zero-fill kernel support (e.g., aclnnInplaceZero*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "ones": [
        "Depends on backend one-fill kernel support (e.g., aclnnInplaceOne*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "ones_like": [
        "Depends on backend one-fill kernel support (e.g., aclnnInplaceOne*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "full": [
        "Depends on backend fill kernel support (e.g., aclnnInplaceFillScalar*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "full_like": [
        "Depends on backend fill kernel support (e.g., aclnnInplaceFillScalar*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "eye": [
        "Depends on backend eye/identity kernel support (e.g., aclnnEye*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "identity": [
        "Depends on backend eye/identity kernel support (e.g., aclnnEye*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
    "linspace": [
        "Depends on backend linspace kernel support (e.g., aclnnLinspace*).",
        "In current implementation, failures are caught by loguru and the API returns None.",
    ],
}


def _env_supports_asnumpy_api(python: str, timeout_s: int) -> bool:
    code = textwrap.dedent(
        """
        import numpy as np
        import asnumpy as ap

        x = ap.zeros((1,), dtype=ap.float32)
        print("ok", x.dtype, x.aclDtype)
        """
    ).strip()
    proc = subprocess.run(
        [python, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode == 0 and (proc.stdout or "").startswith("ok")


def _probe_dtype(python: str, dtype_name: str, expected_acl: int, timeout_s: int) -> dict:
    code = textwrap.dedent(
        f"""
        import json
        import numpy as np
        import asnumpy as ap

        dt = ap.{dtype_name}
        expected_acl = {expected_acl}

        def _ok(arr):
            if arr is None:
                return False
            if int(arr.aclDtype) != expected_acl:
                return False
            if arr.dtype != np.dtype(dt):
                return False
            return True

        supported = []
        results = {{}}

        e = ap.empty((2, 3), dtype=dt)
        assert _ok(e)
        results["empty"] = "supported"
        supported.append("empty")

        candidates = {{
            "zeros": lambda: ap.zeros((2, 3), dtype=dt),
            "ones": lambda: ap.ones((2, 3), dtype=dt),
            "full": lambda: ap.full((2, 3), value=1, dtype=dt),
            "eye": lambda: ap.eye(3, dtype=dt),
            "identity": lambda: ap.identity(3, dtype=dt),
            "linspace": lambda: ap.linspace(0.0, 1.0, steps=6, dtype=dt),
        }}
        for k, fn in candidates.items():
            try:
                arr = fn()
            except Exception as ex:
                results[k] = "error:" + type(ex).__name__
                continue
            if _ok(arr):
                supported.append(k)
                results[k] = "supported"
            else:
                results[k] = "unsupported"

        proto = ap.empty((2, 3), dtype=ap.float32)
        like_candidates = {{
            "zeros_like": lambda: ap.zeros_like(proto, dtype=dt),
            "ones_like": lambda: ap.ones_like(proto, dtype=dt),
            "full_like": lambda: ap.full_like(proto, value=1, dtype=dt),
            "empty_like": lambda: ap.empty_like(proto, dtype=dt),
        }}
        for k, fn in like_candidates.items():
            try:
                arr = fn()
            except Exception as ex:
                results[k] = "error:" + type(ex).__name__
                continue
            if _ok(arr):
                supported.append(k)
                results[k] = "supported"
            else:
                results[k] = "unsupported"

        print(json.dumps({{"dtype": {dtype_name!r}, "expected_acl": expected_acl, "supported": supported, "results": results}}, ensure_ascii=False))
        """
    ).strip()

    proc = subprocess.run(
        [python, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed for {dtype_name} rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads((proc.stdout or "").strip())


def _yaml_escape(s: str) -> str:
    # Keep it simple: always single-quote and escape single quotes.
    return "'" + s.replace("'", "''") + "'"


def _dump_matrix_yaml(matrix: dict) -> str:
    lines: list[str] = []

    lines.append("version: 1")
    lines.append("scope:")
    lines.append("  description: >")
    lines.append("    Support matrix for asnumpy high-level Python APIs vs asnumpy custom dtypes.")
    lines.append("    This focuses on API-level behavior (success / unsupported) rather than registration.")
    lines.append("  environment_notes:")
    lines.append("    - These APIs allocate device memory and require a working Ascend runtime/device.")
    lines.append("    - Some APIs depend on backend kernels and may be unsupported for certain dtypes.")
    lines.append(f"  generated_at_utc: {_yaml_escape(matrix['generated_at_utc'])}")
    lines.append(f"  generated_by: {_yaml_escape(matrix['generated_by'])}")
    lines.append("")

    lines.append("dtypes:")
    for d in matrix["dtypes"]:
        lines.append(f"  - name: {d['name']}")
        lines.append(f"    kind: {d['kind']}")
        lines.append(f"    acl_enum: {d['acl_enum']}")
    lines.append("")

    lines.append("apis:")
    for api in matrix["apis"]:
        lines.append(f"  - name: {api['name']}")
        lines.append(f"    module: {api['module']}")
        lines.append("    status_by_dtype:")
        for dtype_name, status in api["status_by_dtype"].items():
            lines.append(f"      {dtype_name}: {status}")
        if api.get("notes"):
            lines.append("    notes:")
            for note in api["notes"]:
                lines.append(f"      - {note}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update docs/dtype_support_matrix.yaml from live probe.")
    parser.add_argument("--output", default=os.path.join("docs", "dtype_support_matrix.yaml"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()

    if not _env_supports_asnumpy_api(args.python, timeout_s=min(args.timeout_s, 30)):
        print("Environment does not support asnumpy API (no usable Ascend runtime/device). Not updating matrix.")
        return 2

    per_dtype_results: dict[str, dict[str, str]] = {}
    for name, _kind, acl_enum in DTYPES:
        payload = _probe_dtype(args.python, name, acl_enum, timeout_s=args.timeout_s)
        per_dtype_results[name] = payload["results"]

    # Build api list from ordered API_ORDER; only include APIs we probed.
    apis: list[dict] = []
    for api_name in API_ORDER:
        status_by_dtype: dict[str, str] = {}
        for dtype_name, _kind, _acl in DTYPES:
            status_by_dtype[dtype_name] = per_dtype_results[dtype_name].get(api_name, "unknown")
        apis.append(
            {
                "name": api_name,
                "module": "asnumpy.array",
                "status_by_dtype": status_by_dtype,
                "notes": API_NOTES.get(api_name, []),
            }
        )

    matrix = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": os.path.basename(__file__),
        "dtypes": [{"name": n, "kind": k, "acl_enum": a} for (n, k, a) in DTYPES],
        "apis": apis,
    }

    out_text = _dump_matrix_yaml(matrix)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"Updated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

