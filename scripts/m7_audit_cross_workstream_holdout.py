"""E-030: audit E-029 against EVERY supplied historical TRAIN source; refreeze.

Requires the original TRAIN1024 and documented known M5/M6/M7/RSI history.
It does not use dev/eval-A64/final-B160 and runs entirely on CPU.
Do not share raw private holdout/exposure IDs or reviewer packets on GitHub.

Example:
 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python \
 scripts/m7_audit_cross_workstream_holdout.py \
 --frozen <E029_V2> --dataset <ORIGINAL_TRAIN_CONVERTED> \
 --original-train <ORIGINAL_TRAIN1024_JSON> --prior-e026-rows <E026_JSONL> \
 --prior-train-jsonl <M5_ACTIONS> --prior-train-jsonl <M6_ROLLOUT_A> ... \
 --teacher-actions <TEACHER_ACTIONS_TRAIN_JSONL> [repeat for all 6] \
 --optimizer-summary <TRAIN_SUMMARY_JSON> [repeat for all 11] \
 --rsi-solver <VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl> [repeat for three] \
 --rsi-reserved <RSI_HOLDOUT_IDS.jsonl> \
 --obsolete-registry <E029_V1_PRIVATE_REGISTRY.jsonl> \
 --out <NEW_EMPTY_OUTPUT_ROOT>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.holdout_history import make_refreeze


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--original-train", type=Path, required=True)
    p.add_argument("--prior-e026-rows", type=Path, required=True)
    for flag in (
        "prior-train-jsonl", "teacher-actions", "optimizer-summary",
        "rsi-solver", "rsi-reserved", "obsolete-registry",
    ):
        p.add_argument("--"+flag, type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2030)
    a=p.parse_args()
    originals=json.loads(a.original_train.read_text(encoding="utf-8"))
    if (not isinstance(originals,list)
        or len(originals)!=1024
        or any(not isinstance(r,dict) or not isinstance(r.get("_id"),str)
               for r in originals)):
        raise ValueError("Only the pinned original 1024-question TRAIN bank is allowed")
    train_ids={r["_id"] for r in originals}
    if len(train_ids)!=1024:
        raise ValueError("duplicate original TRAIN QID")
    sources = {
        "prior_train_jsonl": a.prior_train_jsonl,
        "teacher_actions": a.teacher_actions,
        "optimizer_summary": a.optimizer_summary,
        "rsi_solver": a.rsi_solver,
        "rsi_reserved": a.rsi_reserved,
        "registry": a.obsolete_registry,
    }
    result=make_refreeze(
        old=a.frozen, valid_train_ids=train_ids, inputs=sources,
        dataset=a.dataset, original_train=a.original_train,
        e026_rows=a.prior_e026_rows, out=a.out, seed=a.seed)
    # Safe aggregate only: do not emit file paths, question IDs or annotations.
    clean = {k:v for k,v in result.items() if k!="historical_source_sha256"}
    print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__=="__main__":
    main()
