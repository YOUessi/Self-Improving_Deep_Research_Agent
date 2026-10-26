"""Matched, nonoracle, untouched TRAIN teacher-reachable *holdout* audit.

Separation from public DEV is intentional: this evaluates known-reachable
examples which were NEVER allowed into the supervised teacher fitting partition
or the online GRPO question selection. The offline TRAIN oracle teacher was
used only to select task IDs. This is *biased TRAIN generalization*, not an
independent external benchmark; cannot promote a checkpoint by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# Support both `python scripts/audit_train_teacher_holdout.py` and
# `python -m scripts.audit_train_teacher_holdout`: direct script execution
# places scripts/ (not its parent) at sys.path[0]. Keep imports explicit.
if __package__:
    from scripts.audit_paired_onpolicy import (
        _json,_read_examples,_check_finite_companion,_bootstrap,_sign_p_improvement,
    )
else:
    from audit_paired_onpolicy import (
        _json,_read_examples,_check_finite_companion,_bootstrap,_sign_p_improvement,
    )

_METRICS=("exact_match","verified_success","evidence_recall",
          "token_f1","tool_calls","search_calls","read_calls")
_BINARY=("exact_match","verified_success")


def audit_holdout(before:Path,after:Path,dataset:Path,
                  trained:Path,out:Path, *,
                  rng_seed:int=20261027,draws:int=5000)->dict:
    if out.exists():raise FileExistsError(out)
    if draws<200:raise ValueError("insufficient bootstrap draws")
    hm=_json(dataset/"TRAIN_HOLDOUT_MANIFEST.json")
    tr=_json(trained/"TRAIN_SUMMARY.json")
    if (hm.get("evaluation_label")!="TRAIN_teacher_success_reachable_holdout_NOT_DEV"
        or hm.get("split")!="train"
        or hm.get("none_in_teacher_fit_partition") is not True
        or hm.get("dev_question_ids_used") is not False
        or hm.get("sealed_final_B160_used") is not False):
        raise ValueError("missing trusted TRAIN-heldout provenance")
    if (tr.get("split")!="train" or tr.get("dev_gold_used_for_reward") is not False
        or tr.get("no_model_oracle_labels") is not True
        or tr.get("sampling_and_loss_distribution_match") is not True
        or tr.get("training_questions_sha256")!=hm.get("original_questions_sha256")
        or tr.get("dev_manifest_sha256")!=hm.get("frozen_public_dev_manifest_sha256")
        or not tr.get("num_updates")):
        raise ValueError("GRPO provenance mismatch or nononline task")
    fit=set(tr["selected_training_ids"])
    targets=set(hm["ids"])
    if fit&targets or len(targets)!=hm["n"]:
        raise ValueError("GRPO trained on heldout TRAIN questions")
    x,y=_json(before/"manifest.json"),_json(after/"manifest.json")
    for m in (x,y):
        if (m.get("split")!="train"
            or m.get("no_weight_updates_in_this_evaluation") is not True
            or m.get("synthetic_fixture")
            or m.get("sample_count")!=hm["n"]
            or m.get("data_sha256",{}).get("documents.jsonl")!=hm["original_documents_sha256"]
            or m.get("data_sha256",{}).get("questions.jsonl")!=hm["holdout_questions_sha256"]):
            raise ValueError("invalid model evaluation or document-IDF mismatch")
    for k in ("data_sha256","seed","agent_config","git_commit","model_id",
              "model_or_policy","sample_count","evaluation_offset",
              "generation_max_new_tokens"):
        if x.get(k)!=y.get(k):
            raise ValueError(f"mismatched holdout paired policy at {k}")
    p=_check_finite_companion(before,after)
    if (p.get("kind")!="finite_constrained_categorical_agent"
        or p.get("answer_proposer")!="v3"
        or p.get("selection_mode")!="sample"
        or p.get("old_adapter_sha256")!=tr.get("old_adapter_sha256")
        or p.get("new_adapter_sha256")!=tr.get("new_adapter_sha256")):
        raise ValueError("checkpoint or actual masked sampling policy mismatch")
    rows0=_read_examples(before/"per_example.jsonl")
    rows1=_read_examples(after/"per_example.jsonl")
    if set(rows0)!=targets or set(rows1)!=targets:
        raise ValueError("holdout question IDs mismatch")
    for qid in targets:
        if rows0[qid].get("split")!="train" or rows1[qid].get("split")!="train":
            raise ValueError("wrong split in model eval")
    rng=random.Random(rng_seed)
    measured={}
    for k in _METRICS:
        b=[float(rows0[q][k]) for q in sorted(targets)]
        a=[float(rows1[q][k]) for q in sorted(targets)]
        if not all(math.isfinite(v) for v in b+a):
            raise ValueError("nonfinite metrics")
        if k in _BINARY and any(v not in (0,1) for v in b+a):
            raise ValueError("invalid binary metrics")
        delta=[v-u for v,u in zip(a,b)]
        result={
            "before_mean":sum(b)/len(b),
            "after_mean":sum(a)/len(a),
            "delta_mean":sum(delta)/len(delta),
            "delta_bootstrap_ci95":_bootstrap(delta,rng=rng,draws=draws),
            "improved_cases":sum(d>0 for d in delta),
            "worsened_cases":sum(d<0 for d in delta),
        }
        if k in _BINARY:
            result["before_count"]=int(sum(b))
            result["after_count"]=int(sum(a))
            result["one_sided_p"]=round(_sign_p_improvement(
                result["improved_cases"],result["worsened_cases"]),7)
        measured[k]=result
    return_record={
        "stage":"TRAIN_TEACHER_REACHABLE_HOLDOUT_PAIRED",
        "n":len(targets),
        "train_fit_question_overlap":0,
        "public_DEV_used_as_optimizer_reward":False,
        "sealed_final_B160_used":False,
        "same_masked_behavior_policy":True,
        "selection_bias":"These TRAIN questions were selected because an offline TRAIN oracle teacher could solve them.",
        "metrics":measured,
        "no_external_generalization_claim":True,
        "eligible_to_promote_checkpoint":False,
        "warning":"Even statistically positive TRAIN teacher-heldout scores cannot replace frozen public DEV/external evidence.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(return_record,indent=2,ensure_ascii=False)+"\n")
    return return_record


def main():
    p=argparse.ArgumentParser()
    for k in ("before","after","dataset","trained","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--draws",type=int,default=5000)
    a=p.parse_args()
    r=audit_holdout(a.before,a.after,a.dataset,a.trained,
                    a.out,draws=a.draws)
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
