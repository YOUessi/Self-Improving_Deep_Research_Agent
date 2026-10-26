"""Paired QA improvement and overhead report without exposing heldout gold rows.

Question IDs must match one-to-one. All measures are per original case,
not averages across unpaired questions. Wilson/bootstrap intervals are
descriptive; small-N significance must not be overstated.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

from sidra.io import iter_jsonl


def _mean(xs):
    return sum(xs)/len(xs)


def _wilson(k,n,z=1.96):
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(max(0,center-margin),5),round(min(1,center+margin),5)]


def compare(baseline:Path, treatment:Path, out:Path,
            *, seed:int=1307, bootstrap:int=5000)->dict:
    if out.exists():raise FileExistsError(out)
    b={r["question_id"]:r for r in iter_jsonl(baseline)}
    t={r["question_id"]:r for r in iter_jsonl(treatment)}
    if not b or b.keys()!=t.keys():
        raise ValueError("paired comparison requires identical nonempty question IDs")
    keys=sorted(b)
    n=len(keys)
    metrics=("verified_success","exact_match","token_f1",
             "evidence_recall","model_calls","prompt_tokens",
             "completion_tokens","tool_calls","elapsed_ms")
    result={"n":n, "paired":True,
            "source":"same frozen question IDs, offline evaluation metrics",
            "seed":seed,"bootstrap_samples":bootstrap,"metrics":{}}
    rng=random.Random(seed)
    for m in metrics:
        vals0=[float(b[k][m] or 0) for k in keys]
        vals1=[float(t[k][m] or 0) for k in keys]
        diffs=[v1-v0 for v0,v1 in zip(vals0,vals1)]
        estimates=[]
        for _ in range(bootstrap):
            estimates.append(sum(diffs[rng.randrange(n)] for __ in range(n))/n)
        estimates.sort()
        s={"baseline_mean":round(_mean(vals0),5),
           "treatment_mean":round(_mean(vals1),5),
           "delta_mean":round(_mean(diffs),5),
           "paired_bootstrap_delta_ci95":[
               round(estimates[int(0.025*bootstrap)],5),
               round(estimates[min(bootstrap-1,int(0.975*bootstrap))],5)],
           "improved_cases":sum(d>1e-10 for d in diffs),
           "worsened_cases":sum(d<-1e-10 for d in diffs)}
        if m in ("verified_success","exact_match"):
            s.update({
                "baseline_count":int(sum(vals0)),
                "treatment_count":int(sum(vals1)),
                "treatment_wilson_ci95":_wilson(int(sum(vals1)),n),
            })
        result["metrics"][m]=s
    result["caveat"]=(
      "These are paired descriptive CIs from frozen public dev, not a guarantee "
      "of generalization to full-Wikipedia, unseen domains or RSI training.")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--treatment",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(compare(a.baseline,a.treatment,a.out),indent=2))


if __name__=="__main__":
    main()
