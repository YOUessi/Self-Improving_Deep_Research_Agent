"""Recompute entire E022->E024 TRAIN source chain; export zero unsafe labels.

Example:
  python scripts/audit_rsi_training_admission.py \
     --train-data ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
     --cohort QUEUE_G1 GROUNDED_G1 SEMANTIC_G1 POSTHOC_G1 \
     --cohort QUEUE_G2 GROUNDED_G2 SEMANTIC_G2 POSTHOC_G2 \
     --cohort QUEUE_G3 GROUNDED_G3 SEMANTIC_G3 POSTHOC_G3 \
     --out-dir ~/experiments/sidra_rsi_m7/trainer_admission/E025_20261027
All reports are source-integrity proofs, NOT positive training labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.rsi_trainer_admission import Cohort,audit_training_admission


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-data",type=Path,required=True)
    parser.add_argument("--cohort",nargs=4,action="append",required=True,
                        metavar=("QUEUE_DIR","GROUNDED_DIR","SEMANTIC_DIR","POSTHOC_DIR"))
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--near-duplicate-cutoff",type=float,default=0.93)
    args=parser.parse_args()
    cohorts=[Cohort(*(Path(p) for p in entry)) for entry in args.cohort]
    result=audit_training_admission(
        args.train_data,cohorts,args.out_dir,
        near_duplicate_cutoff=args.near_duplicate_cutoff,
    )
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":main()
