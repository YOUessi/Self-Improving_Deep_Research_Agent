"""Freeze a label-independent subset of original HotpotQA distractor examples.

Selection is by SHA256(seed:question ID) ascending, never answer correctness.
Outputs a separate JSON with the ORIGINAL format accepted by convert_hotpotqa.py.
The selected examples are local evaluation data; do not commit benchmark contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def select(path: Path, output: Path, *, n: int, seed: int) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    raw_bytes = path.read_bytes()
    rows = json.loads(raw_bytes)
    if not isinstance(rows, list):
        raise ValueError("HotpotQA input must be a list")
    ids = [str(row["_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    ordered = sorted(rows, key=lambda row: (
        hashlib.sha256(f"{seed}:{row['_id']}".encode()).hexdigest(), row["_id"]))
    chosen = ordered[:n]
    if len(chosen) != n:
        raise ValueError("not enough records")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(chosen, ensure_ascii=False), encoding="utf-8")
    manifest = {"source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "source_count": len(rows), "selected_count": len(chosen),
                "seed": seed, "selection": "SHA256(seed:question_id)",
                "selected_ids": [row["_id"] for row in chosen],
                "contains_gold_answers": True,
                "usage": "evaluator-only; never include raw labels in agent prompts"}
    sidecar = output.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    report = select(args.input, args.out, n=args.n, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "selected_ids"}, indent=2))


if __name__ == "__main__":
    main()
