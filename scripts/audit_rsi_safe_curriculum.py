"""Independently audit a frozen TRAIN-only RSI curriculum; no optimizer/GPU.

Outputs only aggregate results, never raw canonical gold/holdout IDs.
Does not open any sealed final-B files or benchmark DEV answer labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.rsi_safe_curriculum_audit import audit_safe_original_train_campaign


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",required=True,type=Path)
    p.add_argument("--public-dev-id-manifest",required=True,type=Path)
    p.add_argument("--history",required=True,type=Path,action="append")
    p.add_argument("--source-sha",required=True)
    p.add_argument("--campaign",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    audit=audit_safe_original_train_campaign(
        args.train_data,args.public_dev_id_manifest,args.history,
        args.campaign,source_sha=args.source_sha,
        config=AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as fp:
        json.dump(audit,fp,ensure_ascii=False,sort_keys=True,indent=2)
        fp.write("\n")
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
