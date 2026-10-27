"""Gold-blind fixed query-rewrite ablation against actual SearchAgent tools.

Predeclared variant: rewrite the already-planned second-hop SEARCH by appending
at most 3 non-duplicate words in the question's interrogative tail.
No target article, answer key or source label is available to either policy.

TRAIN mode is two question-disjoint 64-task cohorts chosen from the same
1,024 frozen TRAIN conversion. The first is the exact M7 seed43 64-task
development sample; the second is fixed seed44 from remaining 960 TRAIN tasks.
EVAL-A64 / final-B160 are neither supported nor loaded. Optional already-open
pilot32 mode requires the exact pinned question and doc SHA *before* reading.
This is scripted retrieval, no model answers or tokens/latency comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from sidra.agent import AgentConfig,SearchAgent
from sidra.bridge_rewrite import rewrite_billed_second_search
from sidra.io import load_dataset,save_jsonl
from sidra.next_hop_policy import plan_next_hop
from scripts.m7_retrieval_probe import (
    ProbePolicy,paired_ci,trace_metrics,
    OPEN_PILOT_DOCS_SHA256,OPEN_PILOT_QUESTIONS_SHA256,
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as s:
        for x in iter(lambda:s.read(1<<20),b""):
            h.update(x)
    return h.hexdigest()


class RelationTailProbePolicy(ProbePolicy):
    """Opt-in, scripted; identical first search/read/finish as frontier-first."""

    def __init__(self):
        super().__init__("frontier_first")
        self.name="m7_frontier_relation_tail_rewrite"
        self.rewrite_actions=0

    def decide(self,question,observations,remaining):
        # Model policies must not react to evaluator/debug events. SearchAgent
        # itself supplies ONLY search/read; filter injected metadata as defence
        # in depth rather than allowing the tail plan to change on gold keys.
        observed=[o for o in observations if o.get("kind") in {"search","read"}]
        reads=[o for o in observed if o.get("kind")=="read"]
        searches=[o for o in observed if o.get("kind")=="search"]
        if (len(reads)==1 and searches
                and observed[-1].get("kind")=="read"):
            plan=plan_next_hop(question,observed,remaining_steps=remaining,
                     remaining_tool_calls=5-len(reads)-len(searches))
            rewritten=rewrite_billed_second_search(plan,question,observed)
            if (plan and rewritten and rewritten.action.kind=="search"
                    and rewritten.action.query!=plan.action.query):
                self.rewrite_actions+=1
                return rewritten.action
        return super().decide(question,observed,remaining)


def sample_train_cohorts(questions:list, *,n:int=64)->dict[str,list]:
    if n<2 or len(questions)<n*2:
        raise ValueError("must have two independently selected TRAIN64 groups")
    if len(set(q.question_id for q in questions))!=len(questions):
        raise ValueError("duplicate questions")
    if any(q.split!="train" for q in questions):
        raise ValueError("TRAIN-only challenge selection")
    ordered=sorted(questions,key=lambda q:q.question_id)
    primary=random.Random(43).sample(ordered,n)
    ids={q.question_id for q in primary}
    remainder=[q for q in ordered if q.question_id not in ids]
    second=random.Random(44).sample(remainder,n)
    assert not ids.intersection(q.question_id for q in second)
    return {
        "train_seed43_development":sorted(primary,key=lambda q:q.question_id),
        "train_seed44_disjoint_holdout":sorted(second,key=lambda q:q.question_id),
    }


def evaluate_cohort(docs,questions,*,pilot:bool=False):
    agent=SearchAgent(docs,ProbePolicy("frontier_first"),
                     AgentConfig(max_steps=6,max_tool_calls=5,
                                 search_top_k=3,max_read_chars=3500))
    variants={
        "frontier_first":ProbePolicy("frontier_first"),
        "relation_tail":RelationTailProbePolicy(),
        "forced_bridge":ProbePolicy("forced_bridge"),
    }
    rows=[{"question_id":q.question_id} for q in questions]
    for variant,policy in variants.items():
        agent.policy=policy
        for q,row in zip(questions,rows):
            before=getattr(policy,"rewrite_actions",0)
            trace=agent.run(q)
            if trace.termination=="error":
                raise RuntimeError(f"{variant} produced illegal tool action: {trace.error}")
            row[variant]=trace_metrics(q,trace)
            row[variant]["relation_rewrite_actions"]=(getattr(policy,"rewrite_actions",0)-before)
    metrics=(
        "retrieval_complete","read_support_complete",
        "retrieval_support_recall","read_support_recall",
        "search_calls","read_calls","tool_calls","relation_rewrite_actions",
    )
    return {
        "n":len(rows),
        "aggregates":{v:{m:round(sum(r[v][m] for r in rows)/len(rows),6)
                         for m in metrics} for v in variants},
        "paired_relation_tail_minus_frontier_first":{
            m:paired_ci(rows,"relation_tail","frontier_first",m,
                        seed=2027,repeats=2000) for m in metrics
        },
        "paired_relation_tail_minus_forced_bridge":{
            m:paired_ci(rows,"relation_tail","forced_bridge",m,
                        seed=2027,repeats=2000) for m in metrics
                        if m!="relation_rewrite_actions"
        },
    },rows


def run(dataset:Path,out:Path,*,mode:str="train",n:int=64)->dict:
    if out.exists():raise FileExistsError(out)
    if mode not in {"train","pilot"}:
        raise ValueError("only TRAIN or previously opened pilot32 allowed")
    if mode=="pilot":
        if (sha256(dataset/"documents.jsonl")!=OPEN_PILOT_DOCS_SHA256
                or sha256(dataset/"questions.jsonl")!=OPEN_PILOT_QUESTIONS_SHA256):
            raise ValueError("not frozen open pilot32")
    docs,questions=load_dataset(dataset)
    if mode=="train":
        cohorts=sample_train_cohorts(questions,n=n)
    else:
        if len(questions)!=32 or any(q.split!="dev" for q in questions):
            raise ValueError("not frozen open pilot32")
        cohorts={"opened_dev_pilot32":sorted(questions,key=lambda q:q.question_id)}
    results={}
    all_rows={}
    for cohort,qs in cohorts.items():
        report,rows=evaluate_cohort(docs,qs,pilot=mode=="pilot")
        results[cohort]=report
        all_rows[cohort]=rows
    report={
        "mode":mode,
        "status":"scripted BM25 SearchAgent retrieval-only, no model calls",
        "data_sha256":{"documents.jsonl":sha256(dataset/"documents.jsonl"),
                       "questions.jsonl":sha256(dataset/"questions.jsonl")},
        "frozen_rewrite_algorithm":(
            "take observed legal frontier-first second SEARCH; append up to "
            "3 relation-tail question words not already in bridge entity/READ title"),
        "train_cohorts_disjoint":mode=="train",
        "no_label_in_search_policy":True,
        "fixed_budget":{"max_steps":6,"max_tool_calls":5,
                        "search_top_k":3,"max_read_chars":3500},
        "cohorts":results,
        "answer_EM":None,
        "original_sentence_citation":None,
        "semantic_entailment":None,
        "model_tokens":None,
        "real_model_latency":None,
        "final_B160_accessed":False,
        "warning":(
            "Single fixed scripted rewrite; never assert model improvement. "
            "Only TRAIN retrieval labels were inspected by the offline scorer, "
            "pilot exports aggregate only. Costs are search/read calls, not LLM latency."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    if mode=="train":
        for cohort,rows in all_rows.items():
            save_jsonl(out/(cohort+"_train_only_metrics.jsonl"),rows)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--mode",choices=("train","pilot"),required=True)
    p.add_argument("--n",type=int,default=64)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.dataset,a.out,mode=a.mode,n=a.n),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
