"""E-027: explicit three-process trust boundary for original-TRAIN Solver research.

prepare: trusted CPU process opens private TRAIN source IDs and documents only,
         exports fresh *model-only* documents with opaque scopes.
solve:   GPU inference process can read ONLY model-only pack; not original QA
         labels, private identity map or frozen DEV metadata.
audit:   independent CPU Verifier opens original TRAIN gold, verifies complete
         BM25/tool replay, then emits public aggregate-only metrics.
No stage here trains a model or automatically approves new generated labels.
"""
import argparse
import json
from pathlib import Path

from sidra.rsi_solver_cohort import (
    audit_solver_run,prepare_solver_pack,run_real_solver,
)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="phase",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--train-documents",type=Path,required=True)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--cohort",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    s=sub.add_parser("solve")
    s.add_argument("--pack",type=Path,required=True)
    s.add_argument("--out",type=Path,required=True)
    s.add_argument("--source-sha",required=True)
    s.add_argument("--endpoint",required=True)
    s.add_argument("--model",required=True)
    s.add_argument("--checkpoint-sha256",required=True)
    s.add_argument("--seed",type=int,default=43)
    s.add_argument("--max-tokens",type=int,default=240)
    a=sub.add_parser("audit")
    a.add_argument("--train-data",type=Path,required=True)
    a.add_argument("--campaign",type=Path,required=True)
    a.add_argument("--pack",type=Path,required=True)
    a.add_argument("--run",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--source-sha",required=True)
    args=parser.parse_args()
    if args.phase=="prepare":
        result=prepare_solver_pack(args.train_documents,args.campaign,args.cohort,
                                   args.out,source_sha=args.source_sha)
    elif args.phase=="solve":
        result=run_real_solver(args.pack,args.out,endpoint=args.endpoint,
                               model=args.model,
                               checkpoint_sha256=args.checkpoint_sha256,
                               source_sha=args.source_sha,seed=args.seed,
                               max_tokens=args.max_tokens)
    else:
        if args.out.exists():raise FileExistsError(args.out)
        result=audit_solver_run(args.train_data,args.campaign,args.pack,args.run,
                                source_sha=args.source_sha)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    # Aggregate/control metadata only; never print TRAIN answer or supporting IDs.
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
