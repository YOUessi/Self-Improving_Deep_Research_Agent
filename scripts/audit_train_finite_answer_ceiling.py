"""TRAIN-only finite-action answer-representation ceiling: 1,024 real QA.

This is a data/algorithm diagnostic. TRAIN gold may select supporting
documents for an *offline hypothetical perfect-READ* condition and SCORE
candidate availability. The normal nonoracle answer proposer receives ONLY
question and purported tool-visible text, never gold strings/labels.

Measures:
 (1) Literal gold answer exists in supporting READ text;
 (2) Actual finite V4 legal Finish catalog offers it;
 (3) Untruncated V2 extractor over original READ text would propose it,
     distinguishing span-extraction limits vs final catalog truncation.

This uses original HotpotQA TRAIN converted, never DEV/eval-A/final-B.
It does NOT train or benchmark an actual agent, and gold-exposure is an
oracle upper bound, not a model-action or retrieval result.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sidra.candidate_proposer import candidate_spans_v2
from sidra.finite_action import FiniteActionConfig,build_legal_actions
from sidra.io import load_dataset
from sidra.metrics import normalize_answer


def evaluate_source_exposure(q, docs_by_id:dict, *,
                             proposer:str="v4")->dict:
    if q.split != "train":
        raise ValueError("only original TRAIN may enter answer ceiling diagnosis")
    if len(set(q.supporting_doc_ids)) < 1:
        raise ValueError("labelled TRAIN support passages required")
    support=[]
    for did in dict.fromkeys(q.supporting_doc_ids):
        d=docs_by_id.get(did)
        if d is None:raise ValueError("TRAIN support passage missing")
        support.append(d)
    obs=[{"kind":"search","query":q.question,
          "hits":[{"doc_id":d.doc_id,"title":d.title,"snippet":d.text[:100]}
                  for d in support]}]
    obs.extend({"kind":"read","doc_id":d.doc_id,"title":d.title,
                "text":d.text[:3500]} for d in support)
    config=FiniteActionConfig(answer_proposer=proposer)
    actions=build_legal_actions(
        q.question,obs,cfg=config,remaining=1,max_tool_calls=5)
    allowed={normalize_answer(a.answer) for a in actions
             if a.kind=="finish" and a.answer.strip()}
    raw=set()
    for d in support:
        raw.update(normalize_answer(x) for x in candidate_spans_v2(
            d.text[:3500],q.question,limit=60))
    gold=normalize_answer(q.gold_answer)
    texts=normalize_answer(" ".join(d.text[:3500] for d in support))
    return {
        "answer_visible":bool(gold and gold in texts),
        "offered_in_final_actions":bool(gold and gold in allowed),
        "offered_in_untruncated_extractor":bool(gold and gold in raw),
        "gold_is_boolean":gold in {"yes","no"},
        "number_of_final_candidates":len(allowed),
        "question_id":q.question_id,
    }


def audit(data:Path,out:Path,*,proposer:str="v4")->dict:
    if out.exists():raise FileExistsError(out)
    if proposer not in {"v3","v4"}:
        raise ValueError("expected frozen V3/V4 finite answer proposer")
    if data.name!="converted" or ("train" not in data.parent.name
                                  and "train" not in str(data).casefold()):
        raise ValueError("this tool requires original TRAIN converted dataset")
    docs,questions=load_dataset(data)
    if not questions or len({q.question_id for q in questions})!=len(questions):
        raise ValueError("empty or duplicate TRAIN questions")
    if any(q.split!="train" for q in questions):
        raise ValueError("never run on DEV/TEST")
    bydoc={d.doc_id:d for d in docs}
    rows=[evaluate_source_exposure(q,bydoc,proposer=proposer)
          for q in questions]
    count=lambda key:sum(int(bool(r[key])) for r in rows)
    n=len(rows)
    hist=Counter(r["number_of_final_candidates"] for r in rows)
    report={
        "stage":"M7A11_FINITE_ANSWER_PROPOSER_REPRESENTATION_CEILING_TRAIN_ONLY",
        "partition":"TRAIN-only; oracle perfect-read upper-bound diagnosis",
        "n":n,
        "proposer":proposer,
        "literal_answer_occurs_in_hypothetical_gold_READ":count("answer_visible"),
        "gold_offered_by_finite_finish_catalog":count("offered_in_final_actions"),
        "gold_offered_by_extended_visible_span_extractor":count(
            "offered_in_untruncated_extractor"),
        "answer_visible_but_final_action_missing":sum(
            r["answer_visible"] and not r["offered_in_final_actions"] for r in rows),
        "extractor_finds_answer_but_finite_catalog_drops":sum(
            r["offered_in_untruncated_extractor"] and
            not r["offered_in_final_actions"] for r in rows),
        "boolean_questions":count("gold_is_boolean"),
        "boolean_finite_catalog_covered":sum(
            r["gold_is_boolean"] and r["offered_in_final_actions"] for r in rows),
        "number_of_final_choices":dict(sorted(hist.items())),
        "uses_train_gold_for_OFFLINE_exposure_and_scoring":True,
        "gold_exposed_to_model_or_online_policy":False,
        "source_search_read_trajectories_are_hypothetical":True,
        "no_weight_update":True,
        "DEV_or_final_B160_used":False,
        "interpretation":(
            "Perfect labelled-source READ is a counterfactual oracle ceiling, "
            "not an Agent result. A correct answer omitted from the finite "
            "Finish choices cannot be learned solely by choice-logit updates."
        ),
    }
    out.parent.mkdir(parents=True,exist_ok=False)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--proposer",choices=["v3","v4"],default="v4")
    a=p.parse_args()
    audit(a.train_data,a.out,proposer=a.proposer)


if __name__=="__main__":
    main()
