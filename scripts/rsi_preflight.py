"""Reproduce gold-redacted, frozen TRAIN-only Challenger queues.

This script ONLY reads a TRAIN corpus and an existing frozen split MANIFEST
(ID lists and hashes, not DEV/TEST questions or labels). It never touches
sealed final-B160 data. No model is loaded and no task is yet generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.experiment import sha256_file
from sidra.io import iter_jsonl,load_dataset
from sidra.rsi_challenge_scheduler import (
    freeze_challenger_queue,select_generation_challenges,stratify_train_questions,
)


def load_denied_ids(frozen_manifest:Path)->set[str]:
    m=json.loads(frozen_manifest.read_text())
    if not all(isinstance(m.get(x),dict) and
               isinstance(m[x].get("selected_ids"),list)
               for x in ("pilot","heldout")):
        raise ValueError("unexpected DEV partition manifest format; fail closed")
    ids=[str(x) for part in ("pilot","heldout")
         for x in m[part]["selected_ids"]]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate protected question IDs")
    return set(ids)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",required=True,type=Path)
    p.add_argument("--frozen-manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--generation",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--size",type=int,default=24)
    p.add_argument("--search-top-k",type=int,default=3)
    p.add_argument("--prior-verifier-map",type=Path)
    p.add_argument("--forbidden-text-sha256",type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    docs,questions=load_dataset(args.train_data)
    denied=load_denied_ids(args.frozen_manifest)
    blocked_hashes=set()
    if args.forbidden_text_sha256:
        blocked_hashes={
            s.strip() for s in args.forbidden_text_sha256.read_text().splitlines()
            if s.strip()}
        if any(len(x)!=64 for x in blocked_hashes):
            raise ValueError("invalid forbidden text hash")
    used_ids=set()
    used_hashes=set()
    if args.prior_verifier_map:
        for row in iter_jsonl(args.prior_verifier_map):
            used_ids.add(str(row["source_train_question_id"]))
            used_hashes.add(str(row["text_sha256"]))
    strata=stratify_train_questions(
        questions,docs,AgentConfig(search_top_k=args.search_top_k),
        forbidden_ids=denied,
        forbidden_normalized_hashes=blocked_hashes)
    selected=select_generation_challenges(
        strata,seed=args.seed,generation=args.generation,
        desired=args.size,prior_source_ids=used_ids,
        prior_text_hashes=used_hashes)
    summary=freeze_challenger_queue(
        selected,args.out,source_git_sha=args.source_sha,
        seed=args.seed,generation=args.generation)
    # Append aggregate-only environment integrity evidence, without
    # copying frozen DEV IDs/questions, original TRAIN gold or model weights.
    summary.update({
        "train_dataset_questions_sha256":sha256_file(
            args.train_data/"questions.jsonl"),
        "train_dataset_documents_sha256":sha256_file(
            args.train_data/"documents.jsonl"),
        "split_manifest_sha256":sha256_file(args.frozen_manifest),
        "reserved_dev_ids_checked_count":len(denied),
        "forbidden_eval_text_fingerprints_checked_count":len(blocked_hashes),
        "train_pool":len(questions),
        "train_retrieval_proxy_pool_counts":{
            name:len(values) for name,values in strata.items()},
        "previous_generation_id_count":len(used_ids),
    })
    (args.out/"QUEUE_MANIFEST.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
