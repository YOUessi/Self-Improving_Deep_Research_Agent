"""Audit RSI TRAIN-only frozen queues and unlabelled Challenger candidates.

Reads ONLY files named CHALLENGER_INPUT.jsonl, VERIFIER_ONLY_MAP.jsonl,
QUEUE_MANIFEST.json and optional candidate summaries. Never opens any DEV,
eval-A or final-B dataset. Emits *aggregate* counts and hashes only.

Usage:
 python scripts/audit_rsi_campaign.py \
   --queues /.../g1_train24_seed43 /.../g2_train24_seed44 /.../g3_train24_seed45 \
   --out /.../reports/train_3g_campaign_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.rsi_campaign_audit import audit_train_campaign


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--queues",type=Path,nargs="+",required=True)
    ap.add_argument("--candidate-dirs",type=Path,nargs="*")
    ap.add_argument("--near-duplicate-threshold",type=float,default=0.92)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("campaign audit already exists; never overwrite")
    report=audit_train_campaign(
        args.queues,candidate_dirs=args.candidate_dirs,
        near_duplicate_threshold=args.near_duplicate_threshold)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
