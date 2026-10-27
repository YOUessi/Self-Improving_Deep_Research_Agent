"""E-026: CPU-only, TRAIN-only curated HotpotQA NLI contrast benchmark.

Executable independent of SearchAgent policy; does NOT consume DEV gold,
modify model weights, alter tool limits, or claim an externally annotated
semantic-entailment ground truth. Official original TRAIN support sentences
are looked up by immutable title + sentence_index keys.

Usage (Tang):
 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 PYTHONPATH=src:. ~/projects/CodeRL-Lab/.venv/bin/python \
 scripts/m7_semantic_contrast_suite.py \
 --dataset <converted_train> --original-train <original_train_subset.json> \
 --model <LOCAL_mDeBERTa_v3_MNLI_snapshot> \
 --cases research/m7_semantic_contrast_v1.json --out <NEW_directory>

Researcher-curated controlled claims, including explicit negation,
unsupported bridge, and removal of one of the original gold support
sentences. An independent human adjudicator has NOT reviewed these labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sidra.semantic_contrast import (
    file_sha256,resolve_controlled_cases,run_controlled_suite,
)
from sidra.semantic_verifier import LocalCPUMNLI


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--original-train",type=Path,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--max-length",type=int,default=384)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):
        raise ValueError("set CUDA_VISIBLE_DEVICES='' for this CPU research job")
    all_cases,meta=resolve_controlled_cases(
        dataset=args.dataset,original_train=args.original_train,
        cases_path=args.cases)
    checkpoints=[p for p in (
        args.model/"model.safetensors",args.model/"pytorch_model.bin")
        if p.is_file()]
    if not checkpoints:
        raise FileNotFoundError("offline local MNLI weights must exist")
    scorer=LocalCPUMNLI(str(args.model),max_length=args.max_length)
    meta["NLI_model"]={
        "local_snapshot":str(args.model),
        "architecture":type(scorer.model).__name__,
        "weights_sha256":file_sha256(checkpoints[0]),
        "label_map":scorer.label_map,
        "max_pair_tokens":scorer.max_length,
        "device":"cpu","web_download":False,
    }
    report=run_controlled_suite(
        all_cases,scorer,meta=meta,out=args.out,bootstrap=args.bootstrap)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
