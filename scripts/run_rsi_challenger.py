"""Run a REAL TRAIN-only local model Challenger; do not award gold labels.

Example:
 python scripts/run_rsi_challenger.py \
  --queue ~/experiments/sidra_rsi_m7/queues/g1_train24_seed43/CHALLENGER_INPUT.jsonl \
  --manifest ~/experiments/sidra_rsi_m7/queues/g1_train24_seed43/QUEUE_MANIFEST.json \
  --endpoint http://127.0.0.1:18985/v1/chat/completions \
  --model sidra-qwen3-8b-awq --out ~/experiments/sidra_rsi_m7/challenger/g1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.rsi_model_challenger import LocalOpenAIChallenger,generate_train_candidates


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--endpoint",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--max-tokens",type=int,default=180)
    args=p.parse_args()
    engine=LocalOpenAIChallenger(
        args.endpoint,args.model,max_tokens=args.max_tokens)
    summary=generate_train_candidates(
        args.queue,args.manifest,args.out,
        backend=engine,seed=args.seed,limit=args.limit)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
