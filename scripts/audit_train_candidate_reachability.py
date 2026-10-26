"""TRAIN-ONLY oracle-source diagnostic of finite-action answer proposer ceilings.

Very important distinctions:
- Gold and supporting_doc_ids are used HERE to construct ideal offline
  source views and calculate upper bounds; this is *not model behavior*.
- The real FiniteActionPolicy never sees gold/supporting_doc_ids. This script
  MUST NEVER be called on DEV/TEST or used as GRPO reward.
- Coverage is only answer *availability* among lexical proposal choices when
  oracle supporting documents are pre-read. It is neither EM nor evidence
  correctness, real retrieval accuracy, RL model quality or RSI.

Reports only aggregate TRAIN statistics, data SHA and ablation config.
No raw answer labels, questions, or full supporting text are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter,defaultdict
from pathlib import Path

from sidra.candidate_proposer import candidate_spans_v2
from sidra.finite_action import _candidate_spans,build_legal_actions,FiniteActionConfig
from sidra.io import load_dataset
from sidra.metrics import normalize_answer
from sidra.retrieval import BM25Index


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _answer_type(question:str)->str:
    text=question.casefold().strip()
    if "how often" in text:return "frequency"
    if re.search(r"\bhow (?:many|much)\b",text):return "quantity"
    if re.search(r"\bwhen\b|\bwhat year\b",text):return "when"
    if re.search(r"\bwhere\b|headquarters",text):return "where"
    if re.search(r"\bwho\b|\bwhom\b|\bwhose\b",text):return "person"
    if re.search(r"\bwhich\b",text):return "which"
    return "other"


def _paired_bootstrap(diff:list[int],*,seed:int=20261027,draws:int=5000)->list[float]:
    n=len(diff)
    rng=random.Random(seed)
    distribution=sorted(sum(diff[rng.randrange(n)] for __ in range(n))/n
                        for _ in range(draws))
    return [round(distribution[int(draws*0.025)],6),
            round(distribution[min(draws-1,int(draws*0.975))],6)]


def audit_train_only(dataset:Path,out:Path,*,seed:int=20261027)->dict:
    if out.exists():raise FileExistsError(out)
    documents,questions=load_dataset(dataset)
    if not questions or any(q.split!="train" for q in questions):
        raise ValueError("oracle-proposer audit is strictly TRAIN-only")
    all_ids=[q.question_id for q in questions]
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("training IDs must be unique")
    docmap={d.doc_id:d for d in documents}
    index=BM25Index(documents)
    overall=Counter()
    by_type=defaultdict(Counter)
    pairdiff=[]
    for q in questions:
        supporting=[]
        for did in q.supporting_doc_ids:
            d=docmap.get(did)
            if d is None:raise ValueError("missing supporting TRAIN document")
            supporting.append(d)
        if not supporting:raise ValueError("missing gold TRAIN support IDs")
        gold=normalize_answer(q.gold_answer)
        if not gold:raise ValueError("TRAIN gold answer missing")
        kind=_answer_type(q.question)
        baseline=any(
            normalize_answer(v)==gold
            for doc in supporting for v in _candidate_spans(doc.text,q.question,limit=3))
        upgraded=any(
            normalize_answer(v)==gold
            for doc in supporting for v in candidate_spans_v2(
                doc.text,q.question,limit=3))
        paired=int(upgraded)-int(baseline)
        pairdiff.append(paired)
        obs=[{"kind":"search","query":"OFFLINE_IDEAL_SUPPORTS_NOT_A_REAL_SEARCH",
              "hits":[{"doc_id":doc.doc_id,"title":doc.title,"snippet":""}
                      for doc in supporting]}]
        obs.extend({
            "kind":"read","doc_id":doc.doc_id,"title":doc.title,
            "text":doc.text[:3500]} for doc in supporting)
        # The ideal trace artificially makes all supporting paragraphs available
        # by their oracle IDs. This never reflects a real policy search success.
        has_gold={}
        for proposer in ("legacy","v2"):
            cfg=FiniteActionConfig(answer_proposer=proposer)
            for remaining in (1,2):
                act=build_legal_actions(
                    q.question,obs,cfg=cfg,remaining=remaining,max_tool_calls=5)
                has_gold[(proposer,remaining)]=any(
                    a.kind=="finish" and normalize_answer(a.answer)==gold
                    for a in act)
        hits=index.search(q.question,k=3,scope=q.scope)
        real_hit_ids={x.doc_id for x in hits}
        features={
            "all":1,
            "gold_mentioned_in_oracle_support":int(any(
                gold in normalize_answer(d.text) for d in supporting)),
            "legacy_candidate_answer_available":int(baseline),
            "v2_candidate_answer_available":int(upgraded),
            "v2_gained":int(upgraded and not baseline),
            "v2_regressed":int(baseline and not upgraded),
            "initial_BM25_top3_any_support":int(bool(real_hit_ids&set(q.supporting_doc_ids))),
            "initial_BM25_top3_all_support":int(set(q.supporting_doc_ids).issubset(real_hit_ids)),
        }
        for (proposer,remaining),flag in has_gold.items():
            features[f"ideal_{proposer}_legal_catalog_step{remaining}_answer_available"]=int(flag)
        overall.update(features)
        by_type[kind].update(features)
    n=overall["all"]
    def _report(counts:Counter)->dict:
        size=counts["all"]
        return {name:{"count":int(v),"fraction":round(v/size,6)}
                for name,v in sorted(counts.items()) if name!="all"}
    result={
        "stage":"TRAIN_ONLY_ORACLE_SUPPORT_DIAGNOSTIC_NOT_A_MODEL_EVAL",
        "train_questions":n,"question_type_subcounts":{
            key:int(group["all"]) for key,group in by_type.items()},
        "train_dataset_sha256":{
            "documents":_sha(dataset/"documents.jsonl"),
            "questions":_sha(dataset/"questions.jsonl"),
        },
        "gold_used_by_actual_agent":False,
        "gold_used_in_offline_diagnostic":True,
        "ideal_support_input_is_oracle_not_retrieval":True,
        "candidate_spans_per_read_doc":3,
        "finite_action_max_candidates":9,
        "original_tool_call_budget":5,
        "metrics":_report(overall),
        "metrics_by_question_type":{
            key:_report(group) for key,group in sorted(by_type.items())},
        "v2_minus_legacy_candidate_answer_fraction":round(
            sum(pairdiff)/n,6),
        "paired_train_bootstrap_delta_ci95":_paired_bootstrap(pairdiff,seed=seed),
        "random_seed":seed,
        "model_learning_demonstrated":False,
        "real_multi_hop_search_demonstrated":False,
        "caveat":(
            "Only TRAIN gold can construct these ideal support views and "
            "measure lexical proposal availability. This is NOT a real "
            "policy rollout, not an independent DEV evaluation, and not "
            "evidence of generalization or GRPO improvement. "
            "Any adoption requires new on-policy TRAIN training plus "
            "same-proposer baseline/after frozen DEV comparisons."
        ),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                   encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261027)
    args=p.parse_args()
    report=audit_train_only(args.train_data,args.out,seed=args.seed)
    print(json.dumps({
        "stage":report["stage"],
        "n":report["train_questions"],
        "legacy":report["metrics"]["legacy_candidate_answer_available"],
        "v2":report["metrics"]["v2_candidate_answer_available"],
        "paired_delta_ci95":report["paired_train_bootstrap_delta_ci95"],
        "model_learning_demonstrated":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
