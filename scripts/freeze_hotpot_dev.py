"""Immutable *dev-only* HotpotQA distractor partitions; no label leakage.

The source is the public HotpotQA distractor DEV JSON, never training data.
Selection is SHA256(seed:question ID) independent of gold answer, support labels,
question difficulty and future model scores. Prior 24 seed42 cases are contained
in pilot32, while heldout224 is disjoint; both remain labeled split='dev'.

Usage:
 python scripts/freeze_hotpot_dev.py --input /data/hotpot_dev_distractor_v1.json \
   --out /experiment/dev256 --source-sha256 <official-sha> \
   --seed 42 --pilot 32 --heldout 224
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from scripts.convert_hotpotqa import convert
else:
    # Invoked directly: sys.path[0] is scripts/, so use sibling import.
    from convert_hotpotqa import convert

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_dev(source: Path, out: Path, *, expected_sha256: str,
               pilot: int = 32, heldout: int = 224, seed: int = 42) -> dict:
    if pilot < 1 or heldout < 1:
        raise ValueError("both partitions must have at least 1 question")
    if out.exists():
        raise FileExistsError(out)
    source_bytes = source.read_bytes()
    observed = _sha(source_bytes)
    if observed != expected_sha256:
        raise ValueError("source SHA256 does not match frozen official dev release")
    rows = json.loads(source_bytes)
    if not isinstance(rows, list):
        raise ValueError("HotpotQA input must be a list")
    if len({str(row["_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate question identifiers")
    ordered = sorted(rows, key=lambda r:
                     (hashlib.sha256(f"{seed}:{r['_id']}".encode()).hexdigest(),
                      str(r["_id"])))
    if len(ordered) < pilot + heldout:
        raise ValueError("not enough questions for frozen partitions")
    # No model sees this answer-bearing original subset. It remains in raw/, while
    # policy inputs only receive Question.question and audited tool observations.
    meta = {"source_sha256": observed, "source_total_questions": len(rows),
            "seed": seed, "selection": "SHA256(seed:question_id) ascending",
            "pilot_size": pilot, "heldout_size": heldout,
            "partition": "pilot=first N hashes, heldout=next K hashes",
            "dataset": "HotpotQA original distractor dev, NOT full Wikipedia",
            "train_eligible": False}
    out.mkdir(parents=True, exist_ok=False)
    for name, chosen in [("pilot", ordered[:pilot]),
                         ("heldout", ordered[pilot:pilot + heldout])]:
        path = out / name
        path.mkdir()
        raw = path / "original_dev_subset.json"
        raw.write_text(json.dumps(chosen, ensure_ascii=False), encoding="utf-8")
        dest = path / "converted"
        convert(raw, dest, "dev")
        (dest / "README.md").write_text(
            f"# Frozen HotpotQA original dev/{name}\n"
            "No portion may be used for train, SFT, GRPO, model selection by gold "
            "labels (heldout), or reward-generation. Scope is per-question "
            "distractor candidate passages, not web retrieval.\n",
            encoding="utf-8")
        meta[name] = {
            "selected_ids": [r["_id"] for r in chosen],
            "original_subset_sha256": _sha(raw.read_bytes()),
            "questions_sha256": _sha((dest / "questions.jsonl").read_bytes()),
            "documents_sha256": _sha((dest / "documents.jsonl").read_bytes()),
            "questions": len(chosen),
        }
    assert not (set(meta["pilot"]["selected_ids"]) & set(meta["heldout"]["selected_ids"]))
    (out / "FROZEN_MANIFEST.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--pilot", type=int, default=32)
    p.add_argument("--heldout", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    meta = freeze_dev(args.input, args.out, expected_sha256=args.source_sha256,
                      pilot=args.pilot, heldout=args.heldout, seed=args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("pilot", "heldout")},
                     indent=2))
    for split in ("pilot", "heldout"):
        print(split, "question_count", meta[split]["questions"],
              "question_sha256", meta[split]["questions_sha256"])


if __name__ == "__main__":
    main()
