"""RSI generation screening: paired DEV pilot, real checkpoint SHA, safe rollback.

The pilot screen cannot by itself promote an RSI checkpoint. A genuine
multi-generation claim additionally requires >=3 independently trained seeds,
new on-policy rollout/reward verification per generation, independent pre-registered
eval-A, and more than one chained improvement. Sealed final-B is NEVER
loaded by this module.

No raw DEV labels/answers/IDs are exported in aggregate reports.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path:Path)->dict:
    result=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result,dict):raise ValueError("expected JSON object")
    return result


def _rows(path:Path)->dict:
    rows={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            qid=row["question_id"]
            if not isinstance(qid,str) or not qid or qid in rows:
                raise ValueError("missing or duplicate paired question IDs")
            rows[qid]=row
    if not rows:raise ValueError("empty eval run")
    return rows


def _paired_ci(deltas:list[float], seed:int, draws:int)->tuple[float,float]:
    if draws<1000:raise ValueError("bootstrap requires >=1000 replicates")
    rng=random.Random(seed)
    n=len(deltas)
    estimates=[]
    for _ in range(draws):
        estimates.append(sum(deltas[rng.randrange(n)] for _ in range(n))/n)
    estimates.sort()
    return (round(estimates[int(0.025*draws)],6),
            round(estimates[min(draws-1,int(0.975*draws))],6))


def screen_paired_dev_pilot(
    before_dir:Path,after_dir:Path,*,seed:int=43,draws:int=4000)->dict:
    """Gold-derived aggregate goes ONLY to evaluator; never to Trainer/Challenger.

    Both runs must be the exact same DEV PILOT QA set, code, search budgets,
    model family and decoding configuration. The only intended variable is
    the LoRA adapter. A single pilot can't certify generalization.
    """
    old=_json(before_dir/"manifest.json")
    new=_json(after_dir/"manifest.json")
    path_old=str(old.get("dataset_dir","")).casefold()
    path_new=str(new.get("dataset_dir","")).casefold()
    for path in (path_old,path_new):
        if (not path or "pilot" not in path
            or any(token in path for token in ("final-b","final_b","b160",
                                               "sealed","/test/","/heldout/"))):
            raise ValueError("evaluation allowed on opened DEV PILOT only")
    fields=(
        "dataset_dir","data_sha256","split","model_id","agent_config",
        "git_commit","sample_count","evaluation_offset","policy",
        "model_or_policy","generation_max_new_tokens",
    )
    for field in fields:
        if old.get(field)!=new.get(field):
            raise ValueError(f"unmatched evaluator policy/config: {field}")
    if old.get("split")!="dev":
        raise ValueError("paired screen refuses TRAIN/TEST evaluation")
    before=_rows(before_dir/"per_example.jsonl")
    after=_rows(after_dir/"per_example.jsonl")
    if set(before)!=set(after):
        raise ValueError("evaluation question IDs differ")
    n=len(before)
    if n<32 or n!=old.get("sample_count"):
        raise ValueError("insufficient/mismatched pilot size")
    question_ids=sorted(before)
    def metric(key:str,*,integer:bool=False):
        left=[float(before[q][key]) for q in question_ids]
        right=[float(after[q][key]) for q in question_ids]
        if not all(math.isfinite(x) for x in left+right):
            raise ValueError("invalid metric")
        if integer and not all(x in (0.0,1.0) for x in left+right):
            raise ValueError("invalid binary metric")
        return left,right,[b-a for a,b in zip(left,right)]
    b_verified,a_verified,d_verified=metric("verified_success",integer=True)
    b_em,a_em,d_em=metric("exact_match",integer=True)
    b_tool,a_tool,d_tool=metric("tool_calls")
    verified_lo,verified_hi=_paired_ci(d_verified,seed,draws)
    em_lo,em_hi=_paired_ci(d_em,seed+1,draws)
    tool_lo,tool_hi=_paired_ci(d_tool,seed+2,draws)
    verified_gain=sum(d_verified)/n
    em_gain=sum(d_em)/n
    tool_increase=sum(d_tool)/n
    def matched_changes(diff):
        return {"improved":sum(d>0 for d in diff),
                "regressed":sum(d<0 for d in diff),
                "unchanged":sum(d==0 for d in diff)}
    reason="no_strict_verified_gain"
    if verified_gain>0 and verified_lo<=0:
        reason="strict_verified_ci_not_positive"
    elif verified_gain>0 and verified_lo>0 and em_gain<=0:
        reason="no_answer_em_gain"
    elif verified_gain>0 and verified_lo>0 and em_gain>0 and tool_increase>0:
        reason="tool_cost_increased"
    elif verified_gain>0 and verified_lo>0 and em_gain>0:
        reason="pilot_candidate_for_independent_eval"
    # Even a strong PILOT result NEVER serves as permission to promote a model.
    candidate=reason=="pilot_candidate_for_independent_eval"
    report={
        "stage":"matched_dev_pilot_screen",
        "n":n,"split":"dev",
        "dataset_sha256":old["data_sha256"],
        "source_git_sha":old.get("git_commit"),
        "decoder_and_budget_matched":True,
        "old_adapter_identity":str(old.get("adapter_path")),
        "new_adapter_identity":str(new.get("adapter_path")),
        "old_manifest_sha256":sha256_file(before_dir/"manifest.json"),
        "new_manifest_sha256":sha256_file(after_dir/"manifest.json"),
        "old_per_example_sha256":sha256_file(before_dir/"per_example.jsonl"),
        "new_per_example_sha256":sha256_file(after_dir/"per_example.jsonl"),
        "paired_bootstrap_replicates":draws,"bootstrap_seed":seed,
        "before_verified_rate":round(sum(b_verified)/n,6),
        "after_verified_rate":round(sum(a_verified)/n,6),
        "paired_verified_delta":round(verified_gain,6),
        "paired_verified_delta_ci95":[verified_lo,verified_hi],
        "verified_paired_changes":matched_changes(d_verified),
        "before_answer_em_rate":round(sum(b_em)/n,6),
        "after_answer_em_rate":round(sum(a_em)/n,6),
        "paired_answer_em_delta":round(em_gain,6),
        "paired_answer_em_delta_ci95":[em_lo,em_hi],
        "answer_paired_changes":matched_changes(d_em),
        "before_tool_calls_mean":round(sum(b_tool)/n,6),
        "after_tool_calls_mean":round(sum(a_tool)/n,6),
        "paired_tool_calls_delta":round(tool_increase,6),
        "paired_tool_calls_delta_ci95":[tool_lo,tool_hi],
        "candidate_for_independent_eval":candidate,
        "screen_reason":reason,
        "promoted":False,
        "requires_independent_eval_A":True,
        "requires_multiple_train_seeds":3,
        "multi_generation_RSI_completed":False,
        "critical_caveat":(
            "DEV PILOT is a tuning screen only, not a valid final model "
            "selection/RSI proof. Aggregate only; never forward per-item "
            "dev labels to Challenger, Solver or Trainer."
        ),
    }
    return report


def make_candidate_record(
    *,generation:int,parent_adapter:Path,candidate_adapter:Path,
    train_rollouts:Path,pilot_report:dict,source_git_sha:str,
    out:Path,previous_record:Path|None=None)->dict:
    """Record real file checksums and a rollback pointer; fail closed.

    If a candidate improves the pilot, keep it pending independent evaluation
    rather than promoting it. If it fails, rollback to the parent checksum;
    retain the actual candidate bytes as a negative experimental artifact.
    """
    if generation<1 or not re.fullmatch(r"[0-9a-f]{40}",source_git_sha):
        raise ValueError("bad generation/source SHA")
    if pilot_report.get("stage")!="matched_dev_pilot_screen":
        raise ValueError("missing independent pilot evaluator report")
    parent=sha256_file(parent_adapter)
    candidate=sha256_file(candidate_adapter)
    rollouts=sha256_file(train_rollouts)
    if candidate==parent:
        raise ValueError("no checkpoint update")
    prev_sha=None
    if generation>1 and previous_record is None:
        raise ValueError("generation >1 must name its predecessor ledger")
    if previous_record:
        previous=_json(previous_record)
        prev_sha=sha256_file(previous_record)
        if previous.get("generation")!=generation-1:
            raise ValueError("broken generation index lineage")
        if parent!=previous.get("selected_for_next_generation_sha256"):
            raise ValueError("parent checkpoint does not match previous rollback/promote pointer")
    candidate_pilot=bool(pilot_report.get("candidate_for_independent_eval"))
    record={
        "generation":generation,
        "state":"PILOT_PENDING_INDEPENDENT_EVAL" if candidate_pilot
                else "REJECTED_ROLLBACK_TO_PARENT",
        "source_git_sha":source_git_sha,
        "previous_record_sha256":prev_sha,
        "parent_checkpoint_sha256":parent,
        "candidate_checkpoint_sha256":candidate,
        "train_rollouts_sha256":rollouts,
        "train_rollouts_filename":train_rollouts.name,
        "source_training_provenance_verified":False,
        "pilot_only_verified_delta":pilot_report["paired_verified_delta"],
        "pilot_only_em_delta":pilot_report["paired_answer_em_delta"],
        "pilot_report_sha256":hashlib.sha256(json.dumps(
            pilot_report,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "pilot_decision":pilot_report["screen_reason"],
        "selected_for_next_generation_sha256":parent,
        "rollback_checkpoint_sha256":parent,
        "independent_eval_A_passed":False,
        "multi_seed_replicated":False,
        "actual_weight_file_differs":True,
        "promoted":False,
        "multi_generation_RSI_completed":False,
        "critical_caveat":(
            "Checksum difference is not proof of a correct Trainer update. "
            "A pilot-only candidate MUST stay unpromoted until independently "
            "validated; negative candidates roll back to the parent."
        ),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:
        json.dump(record,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.write("\n")
    return record


def multi_seed_gate(reports:dict[int,dict])->dict:
    """Gate each generation against multiple actually independent train seeds.

    This does NOT assert independence of examples on a shared DEV pilot.
    """
    seeds=sorted(reports)
    if len(seeds)<3:
        return {"passed":False,"reason":"need_3_independent_train_seeds","seeds":seeds}
    datasets={json.dumps(x.get("dataset_sha256"),sort_keys=True) for x in reports.values()}
    if len(datasets)!=1 or any(x.get("stage")!="matched_dev_pilot_screen"
                             for x in reports.values()):
        return {"passed":False,"reason":"mismatched_eval_datasets_or_stages","seeds":seeds}
    if any(x.get("promoted") for x in reports.values()):
        return {"passed":False,"reason":"pilot_is_not_promotion_proof","seeds":seeds}
    gains=[x["paired_verified_delta"] for x in reports.values()]
    if sum(x>0 for x in gains)<3:
        return {"passed":False,"reason":"some_train_seeds_have_nonpositive_gain","seeds":seeds}
    return {
        "passed":False,
        "reason":"all_pilot_seeds_positive_but_independent_eval_A_still_required",
        "seeds":seeds,
        "pilot_gain_min":min(gains),
        "independent_final_evidence":False,
    }
