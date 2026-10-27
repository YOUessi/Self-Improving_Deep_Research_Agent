"""CPU-only independent re-audit of Qwen Challenger quoted evidence.

No DEV, eval-A, final-B or weight files are accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.rsi_grounded_audit import audit_saved_grounded_candidates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-data",type=Path,required=True)
    ap.add_argument("--queue",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--verifier-map",type=Path,required=True)
    ap.add_argument("--generated",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result=audit_saved_grounded_candidates(
        args.train_data,args.queue,args.manifest,
        args.verifier_map,args.generated)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
