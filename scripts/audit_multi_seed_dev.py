"""Fail-closed multi-seed, question-clustered DEV A/B/C audit.

Every stochastic seed re-evaluates the SAME frozen questions. A 32-question
DEV pilot with two random draws is NOT a 64-question independent test.
Bootstrap resamples QUESTION IDs with all seed outcomes kept together;
no model is promoted by this descriptive, repeatedly inspected public pilot.

Three arms:
  baseline = same frozen starting LoRA
  control  = true on-policy GRPO-only LoRA
  hybrid   = on-policy GRPO + posthoc TRAIN-gold supervised FINISH auxiliary

Check matching source code, same finite masked behavior and tool budgets,
different adapter SHAs, frozen data hashes, and disjoint *evaluation RNG*
seeds. Never imports any sealed final-B160 or training gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

if __package__:
    from scripts.audit_paired_onpolicy import (
        _read_examples,_validate_manifests,METRICS,
    )
else:
    from audit_paired_onpolicy import (
        _read_examples,_validate_manifests,METRICS,
    )


def _hash_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _eval_source(path:Path)->dict:
    return json.loads((path/"manifest.json").read_text())


def _boot_cluster(values:list[float],seed:int,draws:int)->list[float]:
    if len(values)<1 or draws<200:
        raise ValueError("empty question cluster / too few bootstrap draws")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("non-finite paired question differences")
    n=len(values)
    rng=random.Random(seed)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                   for __ in range(draws))
    return [round(samples[int(0.025*draws)],6),
            round(samples[min(draws-1,int(0.975*draws))],6)]


def audit_three_arm_dev(
    runs:list[dict],out:Path,*,bootstrap:int=5000,rng_seed:int=173,
)->dict:
    """Each run: {seed:int, baseline:Path, control:Path, hybrid:Path}.

    Fail closed for nonpaired data, identical adapters, source drift or
    duplicate stochastic seeds; never infer independent sample size across
    stochastic replications of the SAME questions.
    """
    if out.exists():
        raise FileExistsError(out)
    if len(runs)<2:
        raise ValueError("at least two independently-seeded evaluations required")
    if bootstrap<200:
        raise ValueError("at least 200 question-cluster bootstrap draws required")
    dataset_ref=None
    all_seeds=set()
    all_rows={}
    adapter_ref=None
    record=[]
    for entry in runs:
        if set(entry)!={"seed","baseline","control","hybrid"}:
            raise ValueError("three-arm seed record requires exact fields")
        seed=entry["seed"]
        if not isinstance(seed,int) or seed in all_seeds:
            raise ValueError("duplicate or invalid evaluation seed")
        all_seeds.add(seed)
        dirs={role:Path(entry[role]) for role in ("baseline","control","hybrid")}
        if len({str(p.resolve()) for p in dirs.values()})!=3:
            raise ValueError("model arms must be different directories")
        for name in ("control","hybrid"):
            _validate_manifests(dirs["baseline"],dirs[name])
        manifests={k:_eval_source(v) for k,v in dirs.items()}
        if any(m.get("seed")!=seed for m in manifests.values()):
            raise ValueError("declared seed does not match frozen evaluation")
        policies={k:json.loads((v/"FINITE_ACTION_POLICY.json").read_text())
                  for k,v in dirs.items()}
        if any(x.get("selection_mode")!="sample" or
               x.get("answer_proposer")!="v4" or
               x.get("trained_on_eval_gold") is not False for x in policies.values()):
            raise ValueError("requires matched nonoracle v4 sampled behavior")
        weights={k:x["source_adapter_sha256"] for k,x in policies.items()}
        if len(set(weights.values()))!=3:
            raise ValueError("three-arm audit requires 3 distinct adapter SHAs")
        key={
            "data_sha256":manifests["baseline"]["data_sha256"],
            "agent_config":manifests["baseline"]["agent_config"],
            "model_id":manifests["baseline"]["model_id"],
            "git_commit":manifests["baseline"]["git_commit"],
            "evaluation_offset":manifests["baseline"]["evaluation_offset"],
            "sample_count":manifests["baseline"]["sample_count"],
            "answer_proposer":"v4",
        }
        if dataset_ref is None:
            dataset_ref=key
            adapter_ref=weights
        elif key!=dataset_ref or weights!=adapter_ref:
            raise ValueError("seed-to-seed source/data/budget/model/adapter drift")
        rows={k:_read_examples(v/"per_example.jsonl") for k,v in dirs.items()}
        if any(row.keys()!=rows["baseline"].keys() for row in rows.values()):
            raise ValueError("question IDs drift between arms")
        if any(len(row)!=key["sample_count"] for row in rows.values()):
            raise ValueError("manifest/sample count mismatch")
        if all_rows and set(rows["baseline"])!=set(next(iter(all_rows.values()))["baseline"]):
            raise ValueError("seed-to-seed frozen question IDs differ")
        for arm,rs in rows.items():
            for q,row in rs.items():
                if row.get("question_id")!=q or row.get("split")!="dev":
                    raise ValueError("not a genuine frozen public DEV record")
        all_rows[seed]=rows
        record.append({
            "seed":seed,
            "metrics_hash_sha256":{k:_hash_file(v/"per_example.jsonl")
                                  for k,v in dirs.items()},
            "arm_adapter_sha256":weights,
        })
    assert dataset_ref is not None
    ids=sorted(next(iter(all_rows.values()))["baseline"])
    measured={}
    for comp in ("control","hybrid"):
        metrics={}
        for name in METRICS:
            # Keep ALL repeated RNG samples for each question together when
            # computing question-level delta and CI.
            differences=[]
            before_per_question=[]
            after_per_question=[]
            for q in ids:
                b=[];a=[]
                for seed in sorted(all_seeds):
                    b.append(float(all_rows[seed]["baseline"][q][name]))
                    a.append(float(all_rows[seed][comp][q][name]))
                if any(not math.isfinite(v) for v in b+a):
                    raise ValueError(f"non-finite {name}")
                if name in ("exact_match","verified_success"):
                    if any(v not in (0.,1.) for v in b+a):
                        raise ValueError(f"nonbinary {name}")
                before_per_question.append(sum(b)/len(b))
                after_per_question.append(sum(a)/len(a))
                differences.append((sum(a)-sum(b))/len(b))
            avg=sum(differences)/len(ids)
            metrics[name]={
                "baseline_mean":round(sum(before_per_question)/len(ids),6),
                "candidate_mean":round(sum(after_per_question)/len(ids),6),
                "delta_mean":round(avg,6),
                "question_cluster_bootstrap_ci95":_boot_cluster(
                    differences,rng_seed+len(measured)*101+len(metrics),bootstrap),
                "improved_question_count":sum(d>0 for d in differences),
                "worsened_question_count":sum(d<0 for d in differences),
                "independent_question_count":len(ids),
            }
            if name in ("exact_match","verified_success"):
                metrics[name]["counts_per_seed"]={
                    str(seed):{
                        "baseline":int(sum(float(all_rows[seed]["baseline"][q][name])
                                            for q in ids)),
                        "candidate":int(sum(float(all_rows[seed][comp][q][name])
                                             for q in ids)),
                    }
                    for seed in sorted(all_seeds)
                }
        measured[comp]=metrics
    ans=measured
    return_record={
        "study":"M7-A7 three-arm paired question-cluster audit",
        "status":"repeated_public_DEV_pilot_DESCRIPTIVE_only",
        "question_count":len(ids),
        "stochastic_eval_seeds":sorted(all_seeds),
        "total_episodes_per_arm":len(ids)*len(all_seeds),
        "independent_question_count":len(ids),
        "pseudo_replication_prevented":True,
        "source_constraints":dataset_ref,
        "arm_adapter_sha256":adapter_ref,
        "metrics":measured,
        "verified_eval_file_hashes":record,
        "sealed_final_B160_used":False,
        "policy_or_optimizer_exposed_to_DEV_gold":False,
        "any_adapter_promoted":False,
        "independent_generalization_established":False,
        "caution":(
            "Question-clustered CI is descriptive; repeated frozen public DEV "
            "pilot is not a new independent validation set, no matter how many "
            "stochastic evaluation seeds were run. Do not promote checkpoints."
        ),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(return_record,indent=2,ensure_ascii=False)+"\n")
    return return_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",action="append",required=True,
                   help="SEED:BASELINE_DIR:GRPO_CONTROL_DIR:HYBRID_DIR; repeat")
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--rng-seed",type=int,default=173)
    args=p.parse_args()
    runs=[]
    for spec in args.run:
        parts=spec.split(":",3)
        if len(parts)!=4:
            p.error("--run must be SEED:BASELINE:CONTROL:HYBRID")
        try:seed=int(parts[0])
        except ValueError:p.error("invalid numeric --run SEED")
        runs.append(dict(seed=seed,baseline=Path(parts[1]),
                         control=Path(parts[2]),hybrid=Path(parts[3])))
    print(json.dumps(
        audit_three_arm_dev(runs,args.out,bootstrap=args.bootstrap,
                            rng_seed=args.rng_seed),indent=2))


if __name__=="__main__":
    main()
