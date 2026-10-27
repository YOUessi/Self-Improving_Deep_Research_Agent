"""Fail-closed, paired audit for the M7-A8 *disjoint internal TRAIN64* smoke gate.

This is **not** a final benchmark, an independent external sample, nor
an on-policy GRPO reward. Both model outputs must exist before running it.
It never touches eval-A64 or sealed final-B160. Purely evaluator-side
original TRAIN QA labels are used only to score completed test trajectories.

The user-visible report must not re-label supervised hybrids as pure GRPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


MEASURED=("exact_match","verified_success","evidence_recall",
          "tool_calls","read_calls","model_calls","abstained")


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path:Path)->dict:
    row=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row,dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return row


def _rows(path:Path)->dict[str,dict]:
    result={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        row=json.loads(line)
        qid=row.get("question_id")
        if not isinstance(qid,str) or not qid or qid in result:
            raise ValueError(f"duplicate/invalid qid in {path.name}")
        result[qid]=row
    return result


def _ci(values:list[float], *,draws:int,rng:random.Random)->list[float]:
    if len(values)<2 or draws<200:
        raise ValueError("insufficient question clusters or bootstrap draws")
    n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                 for _ in range(draws))
    return [round(means[int(0.025*(draws-1))],8),
            round(means[int(0.975*(draws-1))],8)]


def audit_pair_train64(*,frozen:Path,train_summary:Path,
                       baseline:Path,candidate:Path,out:Path,
                       bootstrap:int=5000,seed:int=919)->dict:
    """Compare same-policy pair; refuses label leakage, data drift, TRAIN fit overlap.

    Checks frozen question-ID membership against EVERY completed optimizer step.
    An optimizer model update without nonconstant TRAIN reward is not accepted.
    """
    if out.exists():
        raise FileExistsError("refusing to overwrite independent TRAIN64 audit")
    hold=_load(frozen/"FREEZE_MANIFEST.json")
    if (hold.get("split")!="train" or hold.get("count")!=64
        or hold.get("model_training_labels_allowed") is not False):
        raise ValueError("invalid frozen, evaluator-only TRAIN64 provenance")
    docsha=_sha256(frozen/"documents.jsonl")
    qsha=_sha256(frozen/"questions.jsonl")
    if (docsha!=hold.get("frozen_documents_sha256")
        or qsha!=hold.get("frozen_questions_sha256")):
        raise ValueError("frozen TRAIN64 dataset changed after preregistration")
    qmap=_rows(frozen/"questions.jsonl")
    if (len(qmap)!=64 or any(q.get("split")!="train" for q in qmap.values())
        or any(not q.get("supporting_doc_ids") for q in qmap.values())):
        raise ValueError("TRAIN64 question labels incomplete or wrong split")
    trainer=_load(train_summary)
    if (trainer.get("split")!="train"
        or trainer.get("dev_gold_used_for_reward") is not False
        or trainer.get("sampling_and_loss_distribution_match") is not True
        or trainer.get("multi_turn_Search_Read_Finish") is not True
        or trainer.get("native_TRL_GRPOTrainer") is not False
        or trainer.get("tool_calls_total") is not None):
        raise ValueError("not a proven on-policy TRAIN-only finite-action experiment")
    if (trainer.get("old_adapter_sha256")==trainer.get("new_adapter_sha256")
        or not all(isinstance(trainer.get(k),str) and len(trainer[k])==64
                   for k in ("old_adapter_sha256","new_adapter_sha256"))):
        raise ValueError("no verifiable checkpoint change")
    updates=trainer.get("updates")
    if (not isinstance(updates,list) or not updates
        or any(not u.get("updated") or u.get("nonconstant_groups",0)<1
               or u.get("behavior_replay_max_logprob_error",1)>1e-3
               or u.get("tool_text_token_targets")!=0 for u in updates)):
        raise ValueError("on-policy update/reward/mask proof missing")
    selected=trainer.get("selected_training_ids")
    if (not isinstance(selected,list) or len(selected)!=len(set(selected))
        or set(selected)&set(qmap)):
        raise ValueError("TRAIN64 leaked into optimizer fit questions")
    runs=[]
    for name,dir_,expect in (
        ("baseline",baseline,trainer["old_adapter_sha256"]),
        ("candidate",candidate,trainer["new_adapter_sha256"]),
    ):
        m=_load(dir_/"manifest.json")
        p=_load(dir_/"FINITE_ACTION_POLICY.json")
        s=_load(dir_/"summary.json")
        rows=_rows(dir_/"per_example.jsonl")
        if (m.get("split")!="train" or m.get("sample_count")!=64
            or s.get("n")!=64 or s.get("split")!="train"):
            raise ValueError(f"{name} run not an exact 64-example TRAIN evaluation")
        if m.get("data_sha256")!={"documents.jsonl":docsha,"questions.jsonl":qsha}:
            raise ValueError(f"{name} evaluation dataset SHA mismatch")
        if (p.get("source_adapter_sha256")!=expect
            or p.get("trained_on_eval_gold") is not False
            or p.get("policy_class")!="finite_constrained_categorical_agent"
            or p.get("selection_mode")!="sample"
            or p.get("answer_proposer")!="v4"):
            raise ValueError(f"{name} adapter/decoder/policy provenance failed")
        if set(rows)!=set(qmap) or any(r.get("split")!="train" for r in rows.values()):
            raise ValueError(f"{name} per-question membership or split mismatch")
        runs.append((m,p,rows))
    m0,p0,a=runs[0]
    m1,p1,b=runs[1]
    for k in ("data_sha256","sample_count","git_commit","model_id",
              "agent_config","seed","evaluation_offset","model_or_policy"):
        if m0.get(k)!=m1.get(k):
            raise ValueError(f"not a matched before/after evaluation: {k}")
    for k in ("answer_proposer","selection_mode","policy_class","tool_budget_max",
              "step_budget_max","categorical_distribution","random_seed_keying"):
        if p0.get(k)!=p1.get(k):
            raise ValueError(f"not a matched legal-action distribution: {k}")
    ids=sorted(qmap)
    rng=random.Random(seed)
    results={}
    for metric in MEASURED:
        x=[a[q].get(metric) for q in ids]
        y=[b[q].get(metric) for q in ids]
        if not all(isinstance(t,(int,float)) and not isinstance(t,bool)
                   and math.isfinite(t) for t in x+y):
            raise ValueError(f"invalid numeric per-example metric: {metric}")
        diffs=[float(v-u) for u,v in zip(x,y,strict=True)]
        results[metric]={
            "baseline_mean":round(sum(x)/len(ids),8),
            "candidate_mean":round(sum(y)/len(ids),8),
            "paired_delta":round(sum(diffs)/len(ids),8),
            "paired_question_bootstrap_ci95":_ci(diffs,draws=bootstrap,rng=rng),
            "better":sum(v>0 for v in diffs),
            "worse":sum(v<0 for v in diffs),
            "tie":sum(v==0 for v in diffs),
        }
    outcome={
        "stage":"M7A8_DISJOINT_TRAIN64_INTERNAL_EVALUATION",
        "n_independent_original_train_questions":64,
        "independent_external_benchmark":False,
        "sealed_final_B160_accessed":False,
        "model_train_gold_used_for_eval":False,
        "frozen_selected_id_sha256":hold.get("ordered_qid_sha256"),
        "frozen_data_sha256":{"questions":qsha,"documents":docsha},
        "trainer_initial_adapter_sha256":trainer["old_adapter_sha256"],
        "trainer_updated_adapter_sha256":trainer["new_adapter_sha256"],
        "train_question_count":len(selected),
        "train_updates":len(updates),
        "same_policy_eval_seed":m0.get("seed"),
        "same_policy_eval_source":m0.get("git_commit"),
        "hybrid_supervision_not_pure_GRPO":trainer.get("any_aux_supervision_is_not_pure_GRPO"),
        "metrics":results,"bootstrap_draws":bootstrap,"bootstrap_rng_seed":seed,
        "promotion_allowed":False,
        "caution":"Disjoint within TRAIN only, not external QA generalization; repeated public DEV pilot is not independent.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n")
    return outcome


def main()->None:
    p=argparse.ArgumentParser()
    for name in ("frozen","train-summary","baseline","candidate","out"):
        p.add_argument("--"+name,required=True,type=Path)
    # Mandatory CLI-only independent source replay. Core audit_pair_train64()
    # remains fixture-testable; production CLI cannot trust the FREEZE_MANIFEST
    # alone or skip historical teacher/optimizer contamination checks.
    p.add_argument("--m7-root",required=True,type=Path,
                   help="M7 root containing the 6 pinned teacher files and 11 prior fits")
    p.add_argument("--original-train-source",required=True,type=Path,
                   help="Original immutable 1024-question TRAIN converted corpus")
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=919)
    args=p.parse_args()
    if __package__:
        from scripts.audit_a8_frozen_integrity import validate_a8_freeze
    else:
        from audit_a8_frozen_integrity import validate_a8_freeze
    provenance=validate_a8_freeze(
        m7root=args.m7_root,source=args.original_train_source,
        frozen=args.frozen)
    audit=audit_pair_train64(
        frozen=args.frozen,train_summary=args.train_summary,
        baseline=args.baseline,candidate=args.candidate,out=args.out,
        bootstrap=args.bootstrap,seed=args.seed)
    # Immutable paired audit already written. A separate stable SHA of the
    # provenance artifact can be kept in the run log without modifying it.
    print(json.dumps({"provenance":provenance,"paired":audit},indent=2))


if __name__=="__main__":
    main()
