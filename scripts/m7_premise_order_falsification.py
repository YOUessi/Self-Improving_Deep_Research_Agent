"""Execute E-028 on pinned E-026 TRAIN-only NLI outputs, CPU ONLY.

Reverses order of exact original TRAIN supporting sentences while preserving
every sentence and the hypothesis. This is a model-order robustness test;
NOT a new semantic ground truth annotation or a new SearchAgent rollout.

Example:
 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 PYTHONPATH=src:. ~/projects/CodeRL-Lab/.venv/bin/python \
 scripts/m7_premise_order_falsification.py \
 --e026-rows .../e026_original_train36_mnli/controlled_rows.jsonl \
 --e026-summary .../e026_original_train36_mnli/summary.json \
 --model <local_cached_mdeberta_mnli_snapshot> --out <NEW_DIR>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sidra.premise_order_audit import (
    run_order_audit,sha256_file,validate_inputs,
)
from sidra.semantic_verifier import LocalCPUMNLI


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--e026-rows",type=Path,required=True)
    p.add_argument("--e026-summary",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--max-length",type=int,default=384)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):
        raise ValueError("CPU-only E-028 requires CUDA_VISIBLE_DEVICES=''")
    summary=json.loads(args.e026_summary.read_text(encoding="utf-8"))
    rows=[json.loads(line) for line in args.e026_rows.read_text(encoding="utf-8").splitlines()]
    eligible=validate_inputs(rows,summary)
    manifest=summary.get("model_manifest")
    if (not isinstance(manifest,dict) or
        manifest.get("device")!="cpu" or
        manifest.get("weights_sha256") is None):
        raise ValueError("missing original E-026 CPU model provenance")
    if args.max_length!=manifest.get("max_pair_tokens"):
        raise ValueError("model token budget differs from original E-026")
    weights=[p for p in (args.model/"model.safetensors",
                          args.model/"pytorch_model.bin") if p.is_file()]
    if len(weights)!=1 or sha256_file(weights[0])!=manifest["weights_sha256"]:
        raise ValueError("NLI checkpoint differs from pinned original E-026")
    scorer=LocalCPUMNLI(str(args.model),max_length=args.max_length)
    result=run_order_audit(rows,summary,scorer,out=args.out,bootstrap=args.bootstrap)
    # Add immutable direct artifacts after successful scoring; fail if
    # output writer changed its output unexpectedly.
    result["source_E026_rows_sha256"]=sha256_file(args.e026_rows)
    result["source_E026_summary_sha256"]=sha256_file(args.e026_summary)
    result["new_NLI_checkpoint_sha256"]=sha256_file(weights[0])
    result["eligible_after_provenance_check"]=len(eligible)
    (args.out/"summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
