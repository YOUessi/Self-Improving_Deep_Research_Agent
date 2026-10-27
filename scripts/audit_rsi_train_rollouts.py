"""Audit saved real TRAIN-only SIDRA trajectories without GPU or DEV gold.

Replays all BM25 hits and read text against original TRAIN corpus, verifies
SearchAgent legality, then recomputes the offline TRAIN-only reward. Neither
trace success nor a matching reward proves actions originated from a model:
model/backend checkpoint provenance is a separate trust requirement.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.experiment import sha256_file
from sidra.io import iter_jsonl,load_dataset
from sidra.retrieval import BM25Index
from sidra.rsi_trace_audit import audit_tool_trace
from sidra.schema import AgentTrace
from sidra.train_reward import score_train_rollout


def verify_train_rollout_file(
    train_data:Path,rollouts:Path,*,config:AgentConfig,
    source_git_sha:str)->dict:
    docs,questions=load_dataset(train_data)
    if any(q.split!="train" for q in questions):
        raise ValueError("TRAIN-only verifier refuses mixed/eval data")
    by_id={q.question_id:q for q in questions}
    dmap={d.doc_id:d for d in docs}
    index=BM25Index(docs,title_boost=config.title_boost)
    count=0;structural=0;finished=0;strict=0;mismatch=0
    reasons=Counter()
    for row in iter_jsonl(rollouts):
        trace=AgentTrace(**row["trace"])
        q=by_id.get(trace.question_id)
        if q is None:
            raise ValueError("rollout contains non-TRAIN question ID")
        if row.get("qid",row.get("question_id"))!=q.question_id:
            raise ValueError("rollout envelope question ID mismatch")
        audit=audit_tool_trace(
            q,trace,docs,config=config,shared_index=index)
        reasons[audit.reason]+=1
        structural+=int(audit.structurally_valid)
        finished+=int(audit.complete_legal_finish)
        verified=score_train_rollout(
            q,trace,dmap,max_tool_calls=config.max_tool_calls)
        actual=verified.as_dict()
        reported=row.get("reward")
        if not isinstance(reported,dict) or set(reported)!=set(actual):
            mismatch+=1
        elif any(abs(float(reported[k])-float(v))>1e-5
                 for k,v in actual.items()):
            mismatch+=1
        # A strict positive requires a legal finished tool trace AND original
        # TRAIN-only independently recomputed answer/evidence reward.
        strict+=int(audit.structurally_valid and
                    audit.complete_legal_finish and verified.strict_outcome==1)
        count+=1
    if not count:raise ValueError("zero TRAIN rollouts")
    return {
        "status":"completed_train_only_offline_verification",
        "source_git_sha":source_git_sha,
        "train_questions_sha256":sha256_file(train_data/"questions.jsonl"),
        "train_documents_sha256":sha256_file(train_data/"documents.jsonl"),
        "rollout_jsonl_sha256":sha256_file(rollouts),
        "n_episodes":count,
        "structurally_valid_tool_traces":structural,
        "legally_completed_finished_answers":finished,
        "original_train_reward_exact_matches":count-mismatch,
        "reported_reward_mismatches":mismatch,
        "independently_validated_strict_successes":strict,
        "reason_histogram":dict(reasons),
        "dev_or_final_b_gold_used":False,
        "model_origin_cryptographically_verified":False,
        "trainer_traces_eligible_only_after_model_origin_check":True,
        "semantic_entailment_proven":False,
        "critical_caveat":"Offline BM25 replay and TRAIN reward recomputation "
                          "do not certify model token origin or semantic entailment; "
                          "failed/abstained traces are not positive task labels.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--rollouts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--max-steps",type=int,default=6)
    p.add_argument("--max-tool-calls",type=int,default=5)
    p.add_argument("--search-top-k",type=int,default=3)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    data=verify_train_rollout_file(
        args.train_data,args.rollouts,
        config=AgentConfig(max_steps=args.max_steps,
                           max_tool_calls=args.max_tool_calls,
                           search_top_k=args.search_top_k),
        source_git_sha=args.source_sha)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(data,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
