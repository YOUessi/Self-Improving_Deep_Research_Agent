"""CPU audit: does the shared evidence progress proxy reward DISTRACTOR reads?

Reuses REAL on-policy TRAIN SearchAgent rollouts saved before M8. This is
offline diagnosis, NOT a new inference/eval or GRPO optimizer update.

The policy never sees gold. We join TRAIN labels only AFTER each true READ
to estimate false-positive state progress on irrelevant documents. If
irrelevant reads frequently gain state potential, this heuristic should NOT
be used for online GRPO until improved.

Only aggregate statistics are exported; no per-question TRAIN gold values
are written to the report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from sidra.evidence_state_credit import assign_evidence_state_credit
from sidra.io import load_dataset,iter_jsonl


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def audit(data:Path,rollouts:Path,out:Path, *,expected_generations:int=4,
          state_version:str="v1")->dict:
    if state_version not in {"v1","v2","v3"}:raise ValueError("unknown state version")
    if out.exists():raise FileExistsError(out)
    if data.name!="converted" or not data.parent.name.startswith("train"):
        raise ValueError("original TRAIN-only data required")
    _,questions=load_dataset(data)
    if any(q.split!="train" for q in questions):
        raise ValueError("DEV/TEST gold cannot enter reward audit")
    qmap={q.question_id:q for q in questions}
    grouped=defaultdict(list)
    rows=list(iter_jsonl(rollouts))
    for r in rows:
        qid=str(r.get("qid",""))
        q=qmap.get(qid)
        if (q is None or r["trace"].get("question_id")!=qid
            or r["trace"].get("question")!=q.question):
            raise ValueError("saved rollout does not match original TRAIN question")
        grouped[(qid,int(r.get("update_iteration",0)))].append(r)
    if not grouped:raise ValueError("empty REAL TRAIN on-policy rollouts")
    total=0
    positive_gold=positive_distractor=0
    reads_gold=reads_distractor=0
    positive_delta_gold=positive_delta_distractor=0.0
    credited=0
    per_kind=defaultdict(int)
    for (qid,update),episodes in sorted(grouped.items()):
        if len(episodes)!=expected_generations:
            raise ValueError("incomplete real TRAIN sibling group")
        q=qmap[qid]
        report=assign_evidence_state_credit(episodes,q,state_version=state_version)
        credited+=report["nonzero_optimizer_action_advantages"]
        for ep in episodes:
            steps=ep["trace"]["steps"]
            for step,sample in zip(steps,ep["action_samples"],strict=True):
                kind=step["action"]["kind"]
                per_kind[kind]+=1
                if kind!="read":continue
                obs=step.get("observation") or {}
                did=obs.get("doc_id")
                if obs.get("kind")!="read" or did!=step["action"]["doc_id"]:
                    raise ValueError("tool history mismatch")
                delta=float(sample["evidence_state_delta_raw"])
                if did in q.supporting_doc_ids:
                    reads_gold+=1
                    positive_gold+=int(delta>1e-8)
                    positive_delta_gold+=delta
                else:
                    reads_distractor+=1
                    positive_distractor+=int(delta>1e-8)
                    positive_delta_distractor+=delta
                total+=1
    report={
        "stage":"M8_REAL_TRAIN_ONPOLICY_ROLLOUT_OBSERVABLE_STATE_PROXY_AUDIT",
        "state_version":state_version,
        "original_TRAIN_question_sha256":digest(data/"questions.jsonl"),
        "real_saved_rollout_sha256":digest(rollouts),
        "episodes":len(rows),"sibling_groups":len(grouped),
        "actual_READ_events":total,
        "actual_gold_READ_events":reads_gold,
        "actual_distractor_READ_events":reads_distractor,
        "positive_state_delta_gold_READ_events":positive_gold,
        "positive_state_delta_distractor_READ_events":positive_distractor,
        "mean_positive_state_delta_for_gold_READ":round(
            positive_delta_gold/max(1,reads_gold),6),
        "mean_positive_state_delta_for_distractor_READ":round(
            positive_delta_distractor/max(1,reads_distractor),6),
        "nonzero_group_relative_action_advantages":credited,
        "action_counts":dict(sorted(per_kind.items())),
        "gold_used_by_policy_or_model_prompt":False,
        "gold_used_only_by_OFFLINE_TRAIN_auditor":True,
        "model_weight_updates":0,
        "risk_note":"Observational potential rises on some distractor reads; compare false-positive reward rate and strict QA before enabling GRPO.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--train-data",type=Path,required=True)
    a.add_argument("--rollouts",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--state-version",choices=("v1","v2","v3"),default="v1")
    s=a.parse_args()
    print(json.dumps(audit(s.train_data,s.rollouts,s.out,
                           state_version=s.state_version),indent=2))


if __name__=="__main__":
    main()
