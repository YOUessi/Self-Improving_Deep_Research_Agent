"""Freeze disjoint HotpotQA TRAIN evidence-adequacy cards for future TWO HUMAN reviewers.

No NLI scoring, no model inference, no grading, no actual annotation: labels
must be supplied independently by human raters after this step. Do NOT upload
raw reviewer packets (containing public corpus text) into project source.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

from sidra.blind_adjudication import freeze_blind_packet


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--original-train",type=Path,required=True)
    p.add_argument("--prior-e026-rows",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=2029)
    args=p.parse_args()
    result=freeze_blind_packet(
        dataset=args.dataset,original_train=args.original_train,
        prior_e026_rows=args.prior_e026_rows,out=args.out,seed=args.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
