"""Calibrate Train-only GRPO reward signal on *actual executed TRAIN traces*.

This is a necessary reward preflight, NOT an RL optimizer run. It reports
whether the reward has any variance and whether successful train trajectories
score above counterfactual broken citations/answers. Counterfactuals are
synthetic perturbations, not newly generated model rollout trajectories.
All gold-labelled inputs MUST have split='train'.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from sidra.io import iter_jsonl,load_dataset
from sidra.schema import AgentTrace
from sidra.train_reward import score_train_rollout


def audit_train_rewards(dataset:Path,traces:Path,out:Path, *,limit:int=100)->dict:
    if out.exists():raise FileExistsError(out)
    docs,qs=load_dataset(dataset)
    qm={q.question_id:q for q in qs}
    lookup={d.doc_id:d for d in docs}
    if any(q.split!="train" for q in qs):
        raise ValueError("DEV/TEST forbidden in TRAIN reward calibration")
    scored=[]
    degraded=[]
    raw=list(iter_jsonl(traces))
    for row in raw[:limit]:
        trace=AgentTrace(**row)
        q=qm.get(trace.question_id)
        if q is None:
            raise ValueError("trace question is not in train data")
        baseline=score_train_rollout(q,trace,lookup)
        scored.append(baseline.total)
        bad=copy.deepcopy(trace)
        bad.citations=bad.citations[:1]
        invalid=score_train_rollout(q,bad,lookup)
        degraded.append(invalid.total)
    if not scored:raise ValueError("no TRAIN traces to calibrate")
    rep={
        "n":len(scored),"source":"TRAIN oracle trajectories only",
        "real_model_rollouts":False,"grpo_optimizer_ran":False,
        "train_only_gold":True,
        "valid_trajectories_reward_mean":round(mean(scored),6),
        "valid_trajectories_reward_min":round(min(scored),6),
        "valid_trajectories_reward_max":round(max(scored),6),
        "synthetic_wrong_citation_reward_mean":round(mean(degraded),6),
        "good_greater_than_corrupt_fraction":round(
            sum(x>y for x,y in zip(scored,degraded))/len(scored),6),
        "sanity_warning":(
            "Synthetic degraded-citation controls are NOT independent model "
            "rollouts; cannot establish GRPO reward variance or effectiveness.")
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rep,indent=2,ensure_ascii=False)+"\n")
    return rep


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--traces",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--limit",type=int,default=100)
    a=p.parse_args()
    print(json.dumps(audit_train_rewards(a.dataset,a.traces,a.out,limit=a.limit),
                     indent=2))


if __name__=="__main__":
    main()
