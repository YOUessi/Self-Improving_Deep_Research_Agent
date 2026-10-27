"""Freeze an untouched TRAIN64 cohort from the ORIGINAL 90,447 TRAIN rows.

IMPORTANT: this is an independent QUESTION cohort drawn from the same
HotpotQA TRAIN distribution, NOT a new benchmark or hidden TEST.

Selection uses only (fixed seed, question ID), NOT answer/support labels.
Excluded: all 1,024 TRAIN questions used for SFT/GRPO/ranker development
(therefore also the history-bound M7 evidence packets from that 1,024),
plus all frozen DEV 256 IDs. This cohort is evaluation-only and MUST be
excluded from any subsequent SFT/GRPO or reward tuning.

Requires original checksum-pinned two TRAIN Parquet shards, pyarrow and
the existing strict HotpotQA converter. Do not commit raw data to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sidra.io import load_dataset
from scripts.build_train_demos import (
    EXPECTED_SHARDS,frozen_train_rows,to_hotpot_original
)
from scripts.convert_hotpotqa import convert


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def choose(rows:list[dict], excluded_ids:set[str], *,n:int=64)->list[dict]:
    if n<1:raise ValueError("must select a nonempty evaluation cohort")
    selected=[]
    seen=set()
    for row in rows:  # rows already in fixed ID-hash ranking
        qid=str(row["id"])
        if qid in seen:raise ValueError("duplicate original TRAIN IDs")
        seen.add(qid)
        if qid in excluded_ids:continue
        selected.append(row)
        if len(selected)>=n:break
    if len(selected)!=n:
        raise ValueError("not enough unexposed TRAIN questions after exclusions")
    return selected


def freeze(*,train_dir:Path,prior_train1024:Path,dev_manifest:Path,
           out:Path,seed:int=71212026,n:int=64)->dict:
    if out.exists():raise FileExistsError(out)
    prior_docs,prior_q=load_dataset(prior_train1024)
    if any(q.split!="train" for q in prior_q):
        raise ValueError("prior source is not TRAIN-only")
    prior_ids={q.question_id for q in prior_q}
    if len(prior_ids)!=len(prior_q) or len(prior_ids)!=1024:
        raise ValueError("expected original complete immutable 1024 TRAIN questions")
    dev=json.loads(dev_manifest.read_text())
    dev_ids=set(dev["pilot"]["selected_ids"])|set(dev["heldout"]["selected_ids"])
    excluded=prior_ids|dev_ids
    # Select original TRAIN Parquet hashes, before consulting any source label.
    paths=[train_dir/name for name in EXPECTED_SHARDS]
    ranked,total,source_hashes=frozen_train_rows(
        paths,n=max(1024,n*5),seed=seed,expected=EXPECTED_SHARDS)
    selected=choose(ranked,excluded,n=n)
    ids=[str(row["id"]) for row in selected]
    assert set(ids).isdisjoint(excluded)
    original=[to_hotpot_original(row) for row in selected]
    out.mkdir(parents=True,exist_ok=False)
    path=out/"original_eval_only_train_subset.json"
    path.write_text(json.dumps(original,ensure_ascii=False))
    convert(path,out/"converted",split="train")
    questions=out/"converted/questions.jsonl"
    documents=out/"converted/documents.jsonl"
    report={
        "stage":"M7_NEW_QUESTION_TRAIN64_EVAL_ONLY",
        "n":len(ids),"full_original_train_count":total,
        "original_train_parquet_sha256":source_hashes,
        "prior_train1024_questions_sha256":sha(prior_train1024/"questions.jsonl"),
        "public_dev_manifest_sha256":sha(dev_manifest),
        "prior_train1024_excluded_count":len(prior_ids),
        "dev_frozen_ids_excluded_count":len(dev_ids),
        "selection":"SHA256(fixed seed:original TRAIN question ID) ascending; labels not consulted",
        "seed":seed,
        "ordered_question_ids_sha256":hashlib.sha256(
            "\n".join(ids).encode()).hexdigest(),
        "converted_questions_sha256":sha(questions),
        "converted_documents_sha256":sha(documents),
        "original_subset_sha256":sha(path),
        "question_id_exclusion_verified":True,
        "optimizer_train_eligible":False,
        "dev_or_hidden_test_data_used":False,
        "research_limit":"Independent question cohort within ORIGINAL HotpotQA TRAIN, not external hidden benchmark. All downstream tuning on these labels prohibited.",
    }
    (out/"FREEZE_MANIFEST.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-dir",type=Path,required=True)
    p.add_argument("--prior-train1024",type=Path,required=True)
    p.add_argument("--dev-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=71212026)
    a=p.parse_args()
    print(json.dumps(freeze(train_dir=a.train_dir,
        prior_train1024=a.prior_train1024,
        dev_manifest=a.dev_manifest,out=a.out,
        seed=a.seed),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
