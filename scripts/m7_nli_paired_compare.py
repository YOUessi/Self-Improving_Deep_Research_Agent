"""Question-cluster paired comparison for TRAIN-only CPU NLI ablations.

NO model/agent inference, no DEV labels, no external network. Input manifests
must be TRAIN-only and have identical source DATA, rollouts and NLI weights.
Output is aggregate only; repeated rollouts of the SAME question are one
bootstrap cluster, never treated as independent questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


METRICS=(
    "nli_scored","model_predicted_entailment",
    "model_predicted_contradiction","answer_em",
    "original_sentence_cited_complete","tool_calls","model_calls",
    "prompt_tokens","completion_tokens","elapsed_ms",
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def read_run(path:Path)->tuple[dict,dict]:
    manifest=json.loads((path/"summary.json").read_text())
    if (manifest.get("stage")!="TRAIN-only offline CPU MNLI experimental screening"
        or manifest.get("no_dev_correct_answer_available_to_verifier") is not True
        or manifest.get("no_eval_A64_or_final_B160_processed") is not True):
        raise ValueError("Refusing non-TRAIN or unverified NLI results")
    rows=[json.loads(x) for x in
          (path/"train_only_per_episode.jsonl").read_text().splitlines() if x.strip()]
    out={}
    for row in rows:
        key=(row["source_experiment"],row["question_id"],row["rollout_index"])
        if key in out:raise ValueError("duplicate ablation episode")
        out[key]=row
    if not out or len(out)!=manifest["n_episodes"]:
        raise ValueError("trace/manifest episode mismatch")
    return manifest,out


def compare(base:Path,candidate:Path,out:Path, *,
            repeats:int=2000,seed:int=2027)->dict:
    if out.exists():raise FileExistsError(out)
    if repeats<100:raise ValueError("bootstrap requires >=100 repeats")
    m0,prior=read_run(base)
    m1,after=read_run(candidate)
    for k in ("source_sha256","NLI_model"):
        if m0[k]!=m1[k]:
            raise ValueError("ablation data/model mismatch: "+k)
    if set(prior)!=set(after):raise ValueError("unpaired episode identities")
    if m0.get("cross_citation_title_link_required") is True:
        raise ValueError("baseline already enforces cross-citation link")
    if m1.get("cross_citation_title_link_required") is not True:
        raise ValueError("candidate must enforce link gate")
    sorted_keys=sorted(prior)
    for key in sorted_keys:
        a,b=prior[key],after[key]
        # Evaluation oracle indicators must be unchanged across guard variants.
        for n in ("answer_em","original_sentence_cited_complete","tool_calls",
                  "model_calls","prompt_tokens","completion_tokens","elapsed_ms"):
            if a[n]!=b[n]:
                raise ValueError(f"replay/gold/cost changed under CPU-only ablation: {n}")
    paired=[(prior[k],after[k]) for k in sorted_keys]
    groups=defaultdict(list)
    for a,b in paired:
        groups[a["question_id"]].append((a,b))
    question_ids=sorted(groups)
    rng=random.Random(seed)
    report={
        "stage":"TRAIN-only question-cluster paired NLI gate ablation",
        "n_episodes":len(paired),"independent_question_clusters":len(groups),
        "bootstrap_repeats":repeats,"seed":seed,
        "baseline_accepts":sum(a["model_predicted_entailment"] for a,_ in paired),
        "candidate_accepts":sum(b["model_predicted_entailment"] for _,b in paired),
        "baseline_wrong_accepts":sum(
            a["model_predicted_entailment"]*(1-a["answer_em"]) for a,_ in paired),
        "candidate_wrong_accepts":sum(
            b["model_predicted_entailment"]*(1-b["answer_em"]) for _,b in paired),
        "baseline_accepts_with_original_sentences":sum(
            a["model_predicted_entailment"]*a["original_sentence_cited_complete"]
            for a,_ in paired),
        "candidate_accepts_with_original_sentences":sum(
            b["model_predicted_entailment"]*b["original_sentence_cited_complete"]
            for _,b in paired),
        "paired_metric_deltas":{},
        "raw_result_sha256":{
            "baseline_summary":sha256(base/"summary.json"),
            "baseline_episode":sha256(base/"train_only_per_episode.jsonl"),
            "candidate_summary":sha256(candidate/"summary.json"),
            "candidate_episode":sha256(candidate/"train_only_per_episode.jsonl"),
        },
        "gold_visible_to_policy":False,
        "meaning":(
            "Only existing TRAIN rollout evaluator labels used for post-hoc "
            "ablation. MNLI positive predictions are NOT entailment truth. "
            "Do NOT promote from 12 question clusters / repeated trajectories."
        ),
    }
    def wrong_rate(a,b):return (b["model_predicted_entailment"]*(1-b["answer_em"])
                                -a["model_predicted_entailment"]*(1-a["answer_em"]))
    def effect(a,b,m):return float(b[m])-float(a[m])
    metrics=list(METRICS)+["wrong_accept"]
    for metric in metrics:
        delta=(lambda a,b:wrong_rate(a,b)) if metric=="wrong_accept" else (
            lambda a,b:effect(a,b,metric))
        actual=sum(delta(a,b) for a,b in paired)/len(paired)
        boots=[]
        for _ in range(repeats):
            sample=[groups[question_ids[rng.randrange(len(question_ids))]]
                    for _ in question_ids]
            size=sum(len(g) for g in sample)
            boots.append(sum(delta(a,b) for group in sample for a,b in group)/size)
        boots.sort()
        report["paired_metric_deltas"][metric]={
            "mean":round(actual,6),
            "ci95":[round(boots[int(0.025*(repeats-1))],6),
                    round(boots[int(0.975*(repeats-1))],6)],
        }
    out.mkdir(parents=True,exist_ok=False)
    (out/"paired_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--bootstrap",default=2000,type=int)
    a=p.parse_args()
    print(json.dumps(compare(a.baseline,a.candidate,a.out,repeats=a.bootstrap),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
