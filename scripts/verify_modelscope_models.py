"""Verify official Qwen3 model files downloaded from ModelScope.

The expected SHA-256 and byte sizes were retrieved from:
  https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Revision=master&Recursive=true
  https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B-AWQ/repo/files?Revision=master&Recursive=true
(2026-10-26). This script does not download or load model weights.

Usage:
  python scripts/verify_modelscope_models.py --base-dir /home/you/models
  python scripts/verify_modelscope_models.py --base-dir /home/you/models --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# (expected bytes, SHA256). SHA-256 is published with the official model files.
EXPECTED: dict[str, dict[str, tuple[int, str]]] = {
    "Qwen3-8B": {
        "model-00001-of-00005.safetensors": (3996250744, "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"),
        "model-00002-of-00005.safetensors": (3993160032, "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"),
        "model-00003-of-00005.safetensors": (3959604768, "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"),
        "model-00004-of-00005.safetensors": (3187841392, "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"),
        "model-00005-of-00005.safetensors": (1244659840, "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"),
        "model.safetensors.index.json": (32878, "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"),
        "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    },
    "Qwen3-8B-AWQ": {
        "model-00001-of-00002.safetensors": (4853922024, "6e112429856bc65e3837a9f38d6f6b71ffdda832cb46299a12f4fa8f6352516e"),
        "model-00002-of-00002.safetensors": (1244659840, "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"),
        "model.safetensors.index.json": (74029, "ad18bc45a440aea0a3298ddfd41d01c48eb9afbd94345f6176c4290e1b524175"),
        "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_model(base: Path, name: str, quick: bool = False) -> dict:
    root = base / name
    files = []
    all_ok = True
    for filename, (size, sha) in EXPECTED[name].items():
        path = root / filename
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else 0
        size_ok = exists and actual_bytes == size
        actual_hash = digest(path) if size_ok and not quick else None
        hash_ok = None if not size_ok or quick else actual_hash == sha
        ok = bool(size_ok and (quick or hash_ok))
        all_ok = all_ok and ok
        files.append({
            "file": filename, "present": exists, "expected_bytes": size,
            "actual_bytes": actual_bytes, "size_match": size_ok,
            "hash_match": hash_ok, "ok": ok,
        })
    missing_metadata = [p for p in ("config.json", "tokenizer_config.json")
                        if not (root / p).is_file()]
    all_ok = all_ok and not missing_metadata
    unfinished = sorted(p.name for p in root.glob("*.incomplete"))
    all_ok = all_ok and not unfinished
    return {
        "model": name, "base_dir": str(root), "sha256_checked": not quick,
        "ready": bool(all_ok), "missing_metadata": missing_metadata,
        "incomplete_files": unfinished, "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path.home() / "models")
    parser.add_argument("--quick", action="store_true",
                        help="Size-only audit; do not treat as final integrity verification")
    parser.add_argument("--model", choices=sorted(EXPECTED), action="append")
    args = parser.parse_args()
    report = [check_model(args.base_dir, m, args.quick)
              for m in args.model or list(EXPECTED)]
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(item["ready"] for item in report) else 2)


if __name__ == "__main__":
    main()
