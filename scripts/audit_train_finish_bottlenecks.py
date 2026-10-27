"""Offline TRAIN-only failure taxonomy for real on-policy finite-action episodes.

Counts separate:
  (1) missing original supporting documents in actually READ observations;
  (2) correct answer unavailable despite complete original evidence;
  (3) correct, fully cited FINISH action available but policy ignored it;
  (4) unexpectedly failed verifier after selecting a correct action;
  (5) strict answer + original-evidence success.

This script diagnoses frozen *training* trajectories; no train-on-DEV
reward, no Oracle model prompts, no sealed final-B160, and no learned model
quality claimed. Do not use public DEV/TEST inputs with this tool.

Run:
 python -m scripts.audit_train_finish_bottlenecks \\
   --train-data <original TRAIN converted> --run <onpolicy output> \\
   --out <new audit path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sidra.io import iter_jsonl
from sidra.metrics import normalize_answer
from sidra.schema import Question


def _hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def classify_train_episode(ep:dict,q:Question)->dict:
    if q.split!="train" or not q.supporting_doc_ids:
        raise ValueError("original TRAIN question with gold supporting IDs required")
    if ep.get("qid")!=q.question_id:
        raise ValueError("trajectory and original TRAIN question ID mismatch")
    tr=ep.get("trace",{})
    if tr.get("question_id")!=q.question_id or tr.get("question")!=q.question:
        raise ValueError("missing original real tool trace")
    steps=tr.get("steps",[])
    samples=ep.get("action_samples",[])
    if not steps or len(steps)!=len(samples):
        raise ValueError("invalid sampled-action/trace alignment")
    reward=ep.get("train_reward",{})
    if any(reward.get(k) not in (0,1,0.,1.)
           for k in ("strict_outcome","answer_correct")):
        raise ValueError("TRAIN reward fields missing or nonbinary")
    answer=normalize_answer(q.gold_answer)
    if not answer:raise ValueError("original TRAIN answer absent")
    gold=set(q.supporting_doc_ids)
    read=set()
    has_answer=False
    has_complete_option=False
    selected_correct=False
    last_selected=None
    for step,sample in zip(steps,samples,strict=True):
        choices=sample.get("legal_actions",[])
        idx=sample.get("selected_index")
        if (not isinstance(idx,int) or not 0<=idx<len(choices)
            or choices[idx]!=step.get("action")
            or sample.get("selection_mode")!="sample"
            or sample.get("action_token_targets")!=1
            or sample.get("tool_observation_targets")!=0):
            raise ValueError("cannot audit mismatched or off-policy ACTION labels")
        chosen=choices[idx]
        for act in choices:
            if act.get("kind")!="finish" or not act.get("answer"):
                continue
            if normalize_answer(str(act["answer"]))==answer:
                has_answer=True
                if gold<=set(act.get("citations",[])):
                    has_complete_option=True
        if (chosen["kind"]=="finish" and chosen.get("answer")
            and normalize_answer(str(chosen["answer"]))==answer):
            selected_correct=True
        last_selected=chosen
        obs=step.get("observation") or {}
        if obs.get("kind")=="read":
            read.add(str(obs.get("doc_id","")))
    if last_selected is None:
        raise ValueError("empty trace")
    strict=bool(reward["strict_outcome"])
    read_complete=gold<=read
    if strict and not (read_complete and has_complete_option and selected_correct
                       and reward["answer_correct"]):
        raise ValueError("claimed strict success conflicts with actual original evidence")
    if strict:
        reason="strict_verified_success"
    elif not read_complete:
        reason="missing_actually_read_original_gold_support"
    elif not has_answer:
        reason="correct_answer_missing_from_action_catalog"
    elif not has_complete_option:
        reason="correct_finish_lacks_full_original_citations"
    elif not selected_correct:
        reason="correct_fully_cited_finish_available_but_not_selected"
    else:
        reason="correct_answer_selected_but_strict_verifier_rejected"
    return {
        "reason":reason,
        "support_fully_read":read_complete,
        "correct_answer_in_action_catalog":has_answer,
        "complete_correct_finish_offered":has_complete_option,
        "correct_answer_finish_selected":selected_correct,
        "strict_verified_success":strict,
        "actual_action_count":len(samples),
    }


def audit_train_finish(data:Path,run:Path,out:Path)->dict:
    if out.exists():raise FileExistsError(out)
    summary=json.loads((run/"TRAIN_SUMMARY.json").read_text(encoding="utf-8"))
    if (summary.get("stage")!="real_online_finite_constrained_categorical_grpo_style"
        or summary.get("split")!="train"
        or summary.get("no_model_oracle_labels") is not True
        or summary.get("dev_gold_used_for_reward") is not False
        or summary.get("sampling_and_loss_distribution_match") is not True
        or not summary.get("num_updates")):
        raise ValueError("not a vetted TRAIN-only real on-policy rollout")
    if _hash(data/"questions.jsonl")!=summary.get("training_questions_sha256"):
        raise ValueError("original TRAIN input checksum mismatch")
    raw=list(iter_jsonl(data/"questions.jsonl"))
    qs={str(q["question_id"]):Question.from_dict(q) for q in raw}
    if len(qs)!=len(raw) or any(q.split!="train" for q in qs.values()):
        raise ValueError("data contains non-TRAIN labels or duplicates")
    allowed=set(summary.get("selected_training_ids",[]))
    if not allowed or not allowed<=qs.keys():
        raise ValueError("invalid selected TRAIN source question IDs")
    cases=Counter()
    observed=Counter()
    n=0
    rows=list(iter_jsonl(run/"train_rollouts.jsonl"))
    expected=sum(int(x["episodes"]) for x in summary["updates"])
    if len(rows)!=expected:
        raise ValueError("training episode count inconsistent with update manifest")
    for ep in rows:
        qid=str(ep.get("qid",""))
        if qid not in allowed:
            raise ValueError("rollout is not from recorded TRAIN optimizer subset")
        cls=classify_train_episode(ep,qs[qid])
        n+=1
        cases[cls["reason"]]+=1
        for field in (
            "support_fully_read","correct_answer_in_action_catalog",
            "complete_correct_finish_offered",
            "correct_answer_finish_selected","strict_verified_success"):
            observed[field]+=int(cls[field])
    report={
        "stage":"TRAIN_ONLY_ONPOLICY_FINISH_BOTTLENECK_DIAGNOSTIC",
        "split":"train","source_train_sha256":_hash(data/"questions.jsonl"),
        "rollout_file_sha256":_hash(run/"train_rollouts.jsonl"),
        "adapter_after_sha256":summary["new_adapter_sha256"],
        "on_policy_distribution_match":True,
        "sealed_final_B160_opened":False,
        "public_DEV_labels_opened":False,
        "n_actual_model_episodes":n,
        "measured_flags":dict(observed),
        "exclusive_failure_taxonomy":dict(cases),
        "no_checkpoint_promotion":True,
        "caution":"Posthoc *TRAIN* diagnostic, task pool may be teacher-selected. "
                  "Action catalog reachability is along the sampled state path, "
                  "NOT full-task learnability or nonoracle success.",
    }
    if sum(cases.values())!=n:raise AssertionError("failed exclusive partition")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",required=True,type=Path)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(audit_train_finish(a.train_data,a.run,a.out),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
