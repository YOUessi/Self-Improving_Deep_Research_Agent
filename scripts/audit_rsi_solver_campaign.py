"""E-028 real, offline, aggregate-only cross-cohort independent TRAIN Solver audit.

No model serving/Optimizer in this process. Never reads public DEV/eval-A/final-B
labels. Re-opens canonical TRAIN source solely inside the independent Verifier.
"""
import argparse
import json
from pathlib import Path

from sidra.rsi_solver_campaign_scorecard import audit_cross_cohort_solver_campaign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--pack", type=Path, action="append", required=True,
                   help="one model-only pack per frozen cohort, in order")
    p.add_argument("--run", type=Path, action="append", required=True,
                   help="one recorded real model-run per cohort, in order")
    p.add_argument("--source-sha", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit_cross_cohort_solver_campaign(
        train_data=args.train_data, campaign=args.campaign, packs=args.pack,
        runs=args.run, source_sha=args.source_sha, out=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
