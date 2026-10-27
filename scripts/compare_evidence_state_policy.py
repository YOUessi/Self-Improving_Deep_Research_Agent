"""Paired evidence-state policy A/B, clustered by original question ID.

Same frozen DEV PILOT32, same model LoRA, same V4 candidate proposer, same
tool budgets, independent two-seed repetitions. The ONLY policy dimension
allowed to differ is evidence_state_mode OFF vs V1/V2/V3.

Report answer/strict citation, gold support READ/SEARCH exposure (POSTHOC
evaluator only), link-state observations and incremental compute cost.

Do not treat 2×32 samples as 64 independent questions, tune on final-B160,
or claim semantic entailment based on titles/lexical links.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
from pathlib import Path

from sidra.evidence_state import derive_evidence_state
from sidra.io import load_dataset,iter_jsonl


METRICS=(
    "exact_match","verified_success","evidence_recall",
    "search_calls","read_calls","tool_calls","model_calls",
    "prompt_tokens","completion_tokens","elapsed_ms",
    "gold_read_pair","gold_search_exposed_pair",
    "observed_cross_links","observed_potential",
)


def load(run:Path,*,expected_seed:int,expected_state:str):
    manifest=json.loads((run/"manifest.json").read_text())
    policy=json.loads((run/"FINITE_ACTION_POLICY.json").read_text())
    if (manifest.get("split")!="dev" or manifest.get("sample_count")!=32
        or manifest.get("seed")!=expected_seed
        or policy.get("answer_proposer")!="v4"
        or policy.get("evidence_state_mode")!=expected_state
        or policy.get("selection_mode")!="sample"):
        raise ValueError("not a matched V4 DEV32 evidence state policy run")
    rows={x["question_id"]:x for x in iter_jsonl(run/"per_example.jsonl")}
    traces={x["question_id"]:x for x in iter_jsonl(run/"traces.jsonl")}
    if len(rows)!=32 or len(traces)!=32 or set(rows)!=set(traces):
        raise ValueError("duplicate/missing traces/metrics")
    return manifest,policy,rows,traces


def compare(pilot:Path,runs:dict[tuple[int,str],Path],out:Path, *,
            seeds:tuple[int,int]=(151,157),bootstrap:int=5000,
            seed:int=831,treatment_mode:str="v1")->dict:
    if treatment_mode not in {"v1","v2","v3"}:
        raise ValueError("treatment state mode must be v1 or v2")
    if out.exists():raise FileExistsError(out)
    if pilot.name!="converted" or pilot.parent.name!="pilot":
        raise ValueError("DEV pilot only; do not inspect frozen eval-A/final-B")
    docs,questions=load_dataset(pilot)
    if len(questions)!=32 or any(q.split!="dev" for q in questions):
        raise ValueError("requires original DEV pilot32")
    qmap={q.question_id:q for q in questions}
    if len(qmap)!=32:raise ValueError("duplicate question IDs")
    if bootstrap<200 or len(set(seeds))!=2:
        raise ValueError("two seeds and >=200 bootstrap draws required")
    data={}
    for s in seeds:
        for mode in ("off",treatment_mode):
            data[(s,mode)]=load(runs[(s,mode)],
                                expected_seed=s,expected_state=mode)
    baseline=data[(seeds[0],"off")]
    baseline_manifest,baseline_policy=baseline[:2]
    for (s,mode),(m,p,rows,traces) in data.items():
        for field in ("git_commit","model_id","agent_config","data_sha256",
                      "adapter_path","sample_count"):
            if m.get(field)!=baseline_manifest.get(field):
                raise ValueError(f"{s} {mode} differs in {field}")
        for field in ("source_adapter_sha256","tool_budget_max",
                      "step_budget_max","answer_proposer",
                      "categorical_distribution"):
            if p.get(field)!=baseline_policy.get(field):
                raise ValueError(f"{s} {mode} differs in policy {field}")
        if set(rows)!=set(qmap):
            raise ValueError("run ids do not match frozen original DEV pilot")
        for qid,trace in traces.items():
            if trace["question"]!=qmap[qid].question:
                raise ValueError("question body drift across eval arms")

    measures={}
    for key in sorted(data):
        s,mode=key
        m,p,rows,traces=data[key]
        scores={}
        for qid in sorted(qmap):
            row=rows[qid]
            trace=traces[qid]
            q=qmap[qid]
            obs=[step["observation"] for step in trace["steps"]
                 if step.get("observation")]
            read_ids={o["doc_id"] for o in obs
                      if o.get("kind")=="read" and o.get("doc_id")}
            exposed_ids={str(h.get("doc_id")) for o in obs
                         if o.get("kind")=="search"
                         for h in o.get("hits",[]) if h.get("doc_id")}
            gold=set(q.supporting_doc_ids)
            state=derive_evidence_state(q.question,obs,version=treatment_mode)
            scored={k:float(row[k]) for k in METRICS
                    if k in row and k not in {
                        "gold_read_pair","gold_search_exposed_pair",
                        "observed_cross_links","observed_potential"}}
            scored.update({
                "gold_read_pair":float(bool(gold) and gold.issubset(read_ids)),
                "gold_search_exposed_pair":float(
                    bool(gold) and gold.issubset(exposed_ids)),
                "observed_cross_links":float(bool(
                    state.linked_pairs if treatment_mode=="v1" else
                    state.anchored_linked_pairs)),
                "observed_potential":state.potential,
            })
            if set(scored)!=set(METRICS) or not all(
                math.isfinite(v) for v in scored.values()):
                raise ValueError("nonfinite or missing paired metrics")
            scores[qid]=scored
        measures[key]=scores
    ids=sorted(qmap)
    rng=random.Random(seed)
    metric_report={}
    for k in METRICS:
        v0=[];v1=[];per=[]
        for qid in ids:
            before=sum(measures[(s,"off")][qid][k] for s in seeds)/len(seeds)
            after=sum(measures[(s,treatment_mode)][qid][k] for s in seeds)/len(seeds)
            v0.append(before);v1.append(after);per.append(after-before)
        dist=sorted(sum(per[rng.randrange(len(ids))] for _ in ids)/len(ids)
                    for _ in range(bootstrap))
        metric_report[k]={
            "off_mean":round(sum(v0)/len(ids),6),
            f"state_{treatment_mode}_mean":round(sum(v1)/len(ids),6),
            "delta":round(sum(per)/len(ids),6),
            "paired_32_question_cluster_bootstrap95":[
                round(dist[int((bootstrap-1)*0.025)],6),
                round(dist[int((bootstrap-1)*0.975)],6)],
            "improved_question_clusters":sum(x>1e-9 for x in per),
            "worsened_question_clusters":sum(x<-1e-9 for x in per),
        }
    out.parent.mkdir(parents=True,exist_ok=True)
    decision=(metric_report["verified_success"]["delta"]>0
              and metric_report["exact_match"]["delta"]>=0)
    result={
        "stage":"M8_GOLD_BLIND_EVIDENCE_STATE_A_B",
        "n_original_question_clusters":len(ids),
        "seeds":list(seeds),
        "episodes_per_arm":len(ids)*len(seeds),
        "split":"already-open DEV pilot32, not external unseen",
        "source_sha":baseline_manifest["git_commit"],
        "adapter_sha":baseline_policy["source_adapter_sha256"],
        "policy":{"answer_proposer":"v4","control":"off",
                  "treatment":treatment_mode,"one_token_categorical":True},
        "metrics":metric_report,
        "state_reconstruction_version_for_both_arms":treatment_mode,
        "state_model_gold_inputs":False,
        "gold_source_labels_used_only_after_trace":True,
        "grant_train_only_online_grpo_trial":decision,
        "decision_rule":"strict delta >0 and answer EM delta >=0, preregistered",
        "science_limits":"Linked-title evidence state is lexical, not entailment. Small reused pilot, not a definitive generalization result.",
    }
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--pilot",type=Path,required=True)
    a.add_argument("--state-version",choices=("v1","v2","v3"),default="v1")
    for seed in (151,157):
        a.add_argument(f"--run-{seed}-off",type=Path,required=True)
        for mode in ("v1","v2","v3"):
            a.add_argument(f"--run-{seed}-{mode}",type=Path)
    a.add_argument("--out",type=Path,required=True)
    v=a.parse_args()
    runs={}
    for seed in (151,157):
        for mode in ("off",v.state_version):
            path=getattr(v,f"run_{seed}_{mode}")
            if path is None:
                a.error(f"--run-{seed}-{mode} is required for {v.state_version}")
            runs[(seed,mode)]=path
    print(json.dumps(compare(v.pilot,runs,v.out,
                             treatment_mode=v.state_version),indent=2))


if __name__=="__main__":
    main()
