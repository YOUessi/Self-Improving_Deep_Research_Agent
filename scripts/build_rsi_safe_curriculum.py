"""Create frozen original-TRAIN task curriculum safely after real M6 trace replay.

THIS IS NOT A NOVEL CHALLENGER OR RSI TRAINING RUN.

Example (Tang CPU, no GPU required):
    PYTHONPATH=src python scripts/build_rsi_safe_curriculum.py \
      --train-data /path/to/TRAIN/converted \
      --public-dev-id-manifest /path/to/FROZEN_MANIFEST.json \
      --history /path/to/M6/runA/rollouts_train.jsonl \
      --history /path/to/M6/runB/rollouts_train.jsonl \
      --source-sha <40char pinned source> --out /new/immutable/campaign
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.rsi_safe_curriculum import build_safe_original_train_campaign


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",required=True,type=Path)
    p.add_argument("--public-dev-id-manifest",required=True,type=Path)
    p.add_argument("--history",required=True,type=Path,action="append")
    p.add_argument("--source-sha",required=True)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--cohorts",type=int,default=3)
    p.add_argument("--per-cohort",type=int,default=24)
    p.add_argument("--train-holdout",type=int,default=32)
    p.add_argument("--search-top-k",type=int,default=3)
    args=p.parse_args()
    result=build_safe_original_train_campaign(
        args.train_data,args.public_dev_id_manifest,args.history,args.out,
        source_sha=args.source_sha,seed=args.seed,
        generations=args.cohorts,per_generation=args.per_cohort,
        holdout_n=args.train_holdout,
        config=AgentConfig(max_steps=6,max_tool_calls=5,
                           search_top_k=args.search_top_k))
    # Never print raw original TRAIN questions, holdout IDs or verifier map.
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
