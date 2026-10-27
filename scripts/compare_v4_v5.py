"""Matched two-seed V4/V5 finite-action Agent comparison (DEV pilot only).

Treat question ID as the bootstrap cluster; the two stochastic seeds are
repeated measurements of the same original 32 questions, NOT 64 independent
examples. Refuse unmatched data, adapter, source SHA or tool budgets.

Run:
 python scripts/compare_v4_v5.py --v4-151 <dir> --v5-151 <dir> \
   --v4-157 <dir> --v5-157 <dir> --out <new report.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


MEASURES=("exact_match","verified_success","token_f1","evidence_recall",
          "search_calls","read_calls","tool_calls","model_calls",
          "prompt_tokens","completion_tokens","elapsed_ms")


def load(run:Path,expected_proposer:str,seed:int)->tuple[dict,dict,dict]:
    m=json.loads((run/"manifest.json").read_text())
    p=json.loads((run/"FINITE_ACTION_POLICY.json").read_text())
    if (p["answer_proposer"]!=expected_proposer or m["seed"]!=seed
        or m["split"]!="dev" or m["sample_count"]!=32
        or p["policy_class"]!="finite_constrained_categorical_agent"
        or p["selection_mode"]!="sample"):
        raise ValueError("unmatched DEV pilot32 finite-action run")
    rows={}
    for line in (run/"per_example.jsonl").read_text().splitlines():
        r=json.loads(line)
        qid=r["question_id"]
        if qid in rows or r["split"]!="dev":
            raise ValueError("duplicate or non-DEV row")
        if any(not isinstance(r[k],(int,float)) or not math.isfinite(r[k])
               for k in MEASURES):
            raise ValueError("non-numeric or nonfinite measured metric")
        rows[qid]=r
    if len(rows)!=32:
        raise ValueError("run does not have exactly 32 QA")
    return m,p,rows


def report_pair(runs:dict[tuple[int,str],Path],out:Path,
                *,seeds:tuple[int,int]=(151,157),bootstrap:int=5000,
                rng_seed:int=157151)->dict:
    if out.exists():raise FileExistsError(out)
    if len(set(seeds))!=2 or bootstrap<200:
        raise ValueError("two seeds and >=200 bootstrap replicates required")
    collected={}
    for seed in seeds:
        for arm in ("v4","v5"):
            collected[(seed,arm)]=load(runs[(seed,arm)],arm,seed)
    reference=collected[(seeds[0],"v4")][0]
    refpolicy=collected[(seeds[0],"v4")][1]
    for (seed,arm),(manifest,policy,rows) in collected.items():
        for key in ("data_sha256","sample_count","git_commit","model_id",
                    "agent_config","adapter_path"):
            if manifest[key]!=reference[key]:
                raise ValueError(f"{seed}/{arm} differs in {key}")
        for key in ("source_adapter_sha256","selection_mode",
                    "tool_budget_max","step_budget_max","categorical_distribution"):
            if policy[key]!=refpolicy[key]:
                raise ValueError(f"{seed}/{arm} differs in policy {key}")
        if set(rows)!=set(collected[(seeds[0],"v4")][2]):
            raise ValueError("question IDs differ between arms/seeds")
    ids=sorted(collected[(seeds[0],"v4")][2])
    rng=random.Random(rng_seed)
    measures={}
    for metric in MEASURES:
        per_id=[]
        base=[]
        cand=[]
        improved=0
        worsened=0
        for qid in ids:
            a=[float(collected[(s,"v4")][2][qid][metric]) for s in seeds]
            b=[float(collected[(s,"v5")][2][qid][metric]) for s in seeds]
            v4=sum(a)/len(a)
            v5=sum(b)/len(b)
            base.append(v4);cand.append(v5)
            d=v5-v4
            per_id.append(d)
            improved+=d>1e-10
            worsened+=d<-1e-10
        draw=sorted(
            sum(per_id[rng.randrange(len(ids))] for _ in ids)/len(ids)
            for _ in range(bootstrap))
        measures[metric]={
            "v4_mean":round(sum(base)/len(ids),6),
            "v5_mean":round(sum(cand)/len(ids),6),
            "v5_minus_v4":round(sum(per_id)/len(ids),6),
            "cluster_bootstrap_95ci":[round(draw[int((bootstrap-1)*0.025)],6),
                                      round(draw[int((bootstrap-1)*0.975)],6)],
            "improved_question_clusters":improved,
            "worsened_question_clusters":worsened,
        }
    result={
        "stage":"M7_V5_REAL_AGENT_TWO_SEED_PAIRED_PILOT32",
        "seed_values":list(seeds),
        "unique_question_id_clusters":len(ids),
        "episodes_per_arm":len(ids)*len(seeds),
        "source_git_commit":reference["git_commit"],
        "source_adapter_sha256":refpolicy["source_adapter_sha256"],
        "data_sha256":reference["data_sha256"],
        "same_model_and_real_search_read_finish":True,
        "same_policy_class_but_different_legal_catalog":True,
        "independent_hidden_test":False,
        "DEV_final_B160_accessed":False,
        "bootstrap_question_clusters":bootstrap,
        "metrics":measures,
        "promote_v5_for_training":(
            measures["verified_success"]["v5_minus_v4"]>0 and
            measures["exact_match"]["v5_minus_v4"]>=0
        ),
        "scope_note":"Same 32 already-exposed DEV questions across two seeds; do NOT interpret as 64 independent QA examples or as a GRPO training gain.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for arm in ("v4","v5"):
        for seed in (151,157):
            p.add_argument(f"--{arm}-{seed}",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    runs={(seed,arm):getattr(a,f"{arm}_{seed}")
          for arm in ("v4","v5") for seed in (151,157)}
    print(json.dumps(report_pair(runs,a.out),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
