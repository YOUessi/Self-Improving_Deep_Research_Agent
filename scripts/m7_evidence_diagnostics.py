"""Reproducible M7 evidence bottleneck / non-oracle verifier attribution.

TRAIN mode: only archived TRAIN rollouts + exact TRAIN original HotpotQA
(title,sentence_index) labels; computes OFFLINE post-episode causally cautious
bottleneck frequencies. Cluster bootstrap resamples QUESTIONS (not correlated
rollouts), never treats 4 generations from one task as four independent cases.

PILOT mode: checks the already-open pilot32 dataset SHA256 in an existing
evaluation manifest BEFORE reading trajectories. Computes observable-only
rejection reasons, token and tool costs; does NOT open DEV questions, answer
keys, supporting passages, sentence facts, eval-A64 or sealed final-B160.
No semantic NLI model is invoked; semantics remain UNVERIFIED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

from sidra.evidence_diagnostics import offline_train_diagnosis,source_only_verdict
from sidra.io import iter_jsonl,load_dataset,save_jsonl
from scripts.m7_retrieval_probe import (
    OPEN_PILOT_DOCS_SHA256,OPEN_PILOT_QUESTIONS_SHA256,
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(1<<20),b""):
            h.update(blob)
    return h.hexdigest()


def _read_json_array(path:Path)->list[dict]:
    j=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(j,list) or not all(isinstance(i,dict) for i in j):
        raise ValueError("original dataset must be a JSON array")
    return j


def clustered_bootstrap(rows:list[dict], fields:tuple[str,...], *,
                        seed:int=2027,repeats:int=2000)->dict:
    if not rows or repeats<100:
        raise ValueError("need nonempty episodes and >=100 bootstrap draws")
    groups:dict[str,list[dict]]=defaultdict(list)
    for r in rows:
        groups[str(r["question_id"])].append(r)
    keys=sorted(groups)
    rnd=random.Random(seed)
    report={}
    for field in fields:
        vals=[float(r[field]) for r in rows]
        samples=[]
        for _ in range(repeats):
            picks=[groups[keys[rnd.randrange(len(keys))]] for _ in keys]
            nums=[float(row[field]) for block in picks for row in block]
            samples.append(sum(nums)/len(nums))
        samples.sort()
        report[field]={
            "mean":round(sum(vals)/len(vals),6),
            "ci95_question_cluster_bootstrap":[
                round(samples[int(0.025*repeats)],6),
                round(samples[min(repeats-1,int(0.975*repeats))],6),
            ],
            "question_clusters":len(keys),
            "episodes":len(rows),
        }
    return report


TRAIN_FIELDS=(
    "answer_em","support_retrieval_complete","support_read_complete",
    "support_passage_cited_complete","original_sentence_read_complete",
    "original_sentence_cited_complete","answer_and_original_sentence_cited",
    "read_sentence_fraction","cited_sentence_fraction",
    "source_lexical_candidate","source_cross_document_phantom_candidate",
    "support_retrieval_recall","support_read_recall",
    "tool_calls","model_calls","prompt_tokens","completion_tokens","elapsed_ms",
)
PILOT_FIELDS=("lexical_candidate","concat_only_lexical_candidate",
              "error","abstained","tool_calls","model_calls",
              "prompt_tokens","completion_tokens","elapsed_ms")


def run_train(data:Path,original:Path,rollouts:Path,out:Path, *,
              bootstrap:int=2000)->dict:
    if out.exists():raise FileExistsError(out)
    docs,qs=load_dataset(data)
    if any(q.split!="train" for q in qs):raise ValueError("TRAIN-only mode: DEV/test denied")
    if not docs:raise ValueError("TRAIN document corpus empty")
    originals=_read_json_array(original)
    qmap={q.question_id:q for q in qs}
    rawmap={r.get("_id"):r for r in originals}
    if (len(rawmap)!=len(originals) or len(qmap)!=len(originals)
            or set(qmap)!=set(rawmap)):
        raise ValueError("original TRAIN subset and converted TRAIN IDs must match exactly")
    for qid,q in qmap.items():
        if q.question!=rawmap[qid].get("question"):
            raise ValueError("original and converted TRAIN questions disagree")
    episodes=list(iter_jsonl(rollouts))
    if not episodes:raise ValueError("empty TRAIN rollouts")
    keys=set()
    rows=[]
    for ep in episodes:
        qid=ep.get("question_id")
        idx=ep.get("rollout_index")
        if qid not in qmap:raise ValueError("trace not present in TRAIN corpus")
        if not isinstance(idx,int) or idx<0 or (qid,idx) in keys:
            raise ValueError("duplicate/missing rollout index within a TRAIN question")
        keys.add((qid,idx))
        trace=ep.get("trace")
        if not isinstance(trace,dict) or trace.get("question_id")!=qid:
            raise ValueError("rollout and nested trace question mismatch")
        row=offline_train_diagnosis(qmap[qid],trace,rawmap[qid])
        row["rollout_index"]=idx
        row["reward_total"]=float(ep.get("reward",{}).get("total",0))
        rows.append(row)
    ci=clustered_bootstrap(rows,TRAIN_FIELDS,seed=2027,repeats=bootstrap)
    report={
        "mode":"train",
        "n_train_questions":len({r["question_id"] for r in rows}),
        "n_episodes":len(rows),
        "source_dataset_questions":len(qs),
        "input_sha256":{
            "questions.jsonl":sha256(data/"questions.jsonl"),
            "documents.jsonl":sha256(data/"documents.jsonl"),
            "original_train_subset.json":sha256(original),
            "train_rollouts.jsonl":sha256(rollouts),
        },
        "primary_bottleneck_counts":dict(sorted(Counter(
            r["primary_evidence_bottleneck"] for r in rows).items())),
        "observable_source_verdict_counts":dict(sorted(Counter(
            r["source_reason"] for r in rows).items())),
        "termination_counts":dict(sorted(Counter(r["termination"] for r in rows).items())),
        "question_cluster_bootstrap":ci,
        "bootstrap_seed":2027,
        "bootstrap_repeats":bootstrap,
        "answer_EM_from_train_labels":True,
        "original_hotpot_supporting_facts":True,
        "semantic_entailment_evaluated":False,
        "no_dev_or_final_labels_accessed":True,
        "interpretation":(
            "Real archived model SearchAgent TRAIN rollouts: offline support "
            "passage/sentence exposure, answer EM, observed source-only "
            "verdict and cost. Bottleneck priority is descriptive, not causal. "
            "Lexical candidate and official sentence presence DO NOT imply entailment."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    save_jsonl(out/"train_only_per_episode.jsonl",rows)
    return report


def run_pilot(traces:Path,manifest:Path,out:Path, *,
              bootstrap:int=2000)->dict:
    if out.exists():raise FileExistsError(out)
    # This check MUST happen before reading the trajectories. Never access
    # hidden dev labels via random-DEV32 or an eval-A / sealed final manifest.
    m=json.loads(manifest.read_text(encoding="utf-8"))
    sha=m.get("data_sha256") or {}
    if not (m.get("split")=="dev" and m.get("sample_count")==32
            and sha.get("documents.jsonl")==OPEN_PILOT_DOCS_SHA256
            and sha.get("questions.jsonl")==OPEN_PILOT_QUESTIONS_SHA256):
        raise ValueError("not the previously opened frozen pilot32 manifest")
    episodes=list(iter_jsonl(traces))
    ids=[ep.get("question_id") for ep in episodes]
    if len(ids)!=32 or len(set(ids))!=32:
        raise ValueError("pilot requires exactly 32 distinct question traces")
    rows=[]
    for tr in episodes:
        source=source_only_verdict(tr)
        rows.append({
            "question_id":tr["question_id"],
            "source_reason":source.reason,
            "lexical_candidate":int(source.lexical_candidate),
            "concat_only_lexical_candidate":int(source.concat_only_lexical_candidate),
            "error":int(tr.get("termination")=="error"),
            "abstained":int(tr.get("termination")=="abstained"),
            "tool_calls":int(tr.get("tool_calls",0)),
            "model_calls":int(tr.get("model_calls",0)),
            "prompt_tokens":int(tr.get("prompt_tokens",0)),
            "completion_tokens":int(tr.get("completion_tokens",0)),
            "elapsed_ms":int(tr.get("elapsed_ms",0)),
        })
    report={
        "mode":"opened_DEV_pilot32_observations_only",
        "n":len(rows),
        "existing_run_policy":m.get("model_or_policy"),
        "source_observable_reason_counts":dict(sorted(Counter(
            r["source_reason"] for r in rows).items())),
        "question_bootstrap":clustered_bootstrap(
            rows,PILOT_FIELDS,seed=2027,repeats=bootstrap),
        "input_sha256":{"traces.jsonl":sha256(traces),
                        "manifest.json":sha256(manifest)},
        "pilot_dataset_sha256":sha,
        "bootstrap_seed":2027,"bootstrap_repeats":bootstrap,
        "heldout_aggregate_only":True,
        "answer_em":None,
        "gold_sentence_exposure":None,
        "semantic_entailment":None,
        "note":(
            "Only previously opened pilot32 run and actual SEARCH/READ "
            "observations were read. No DEV label, original sentence fact, "
            "eval-A64 or sealed final-B160. No per-question report exported."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("train","pilot"),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--rollouts",type=Path)
    p.add_argument("--traces",type=Path)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--original-train",type=Path)
    p.add_argument("--pilot-manifest",type=Path)
    p.add_argument("--bootstrap",type=int,default=2000)
    a=p.parse_args()
    if a.mode=="train":
        if a.rollouts is None or a.dataset is None or a.original_train is None:
            p.error("train requires --rollouts --dataset --original-train")
        report=run_train(a.dataset,a.original_train,a.rollouts,a.out,
                         bootstrap=a.bootstrap)
    else:
        if a.traces is None or a.pilot_manifest is None:
            p.error("pilot requires --traces --pilot-manifest")
        report=run_pilot(a.traces,a.pilot_manifest,a.out,bootstrap=a.bootstrap)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
