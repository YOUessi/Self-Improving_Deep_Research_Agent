"""Select a title-aware BM25 coefficient using PILOT dev only.

Do NOT search the held-out 224 labels during parameter tuning. This code
produces a frozen candidate hyperparameter (not an Agent result).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from sidra.io import load_dataset
from sidra.retrieval import BM25Index


def score_retriever(data: Path, boost: float, *, topk=(3,5)) -> dict:
    docs, questions = load_dataset(data)
    if any(q.split != "dev" for q in questions):
        raise ValueError("retrieval tuning requires public dev-only questions")
    idx = BM25Index(docs,title_boost=boost)
    found = {k:0 for k in topk}
    fraction = {k:0.0 for k in topk}
    for q in questions:
        gold = set(q.supporting_doc_ids)
        hits = idx.search(q.question,scope=q.scope,k=max(topk))
        for k in topk:
            ids = {h.doc_id for h in hits[:k]}
            found[k] += int(bool(gold) and gold.issubset(ids))
            fraction[k] += len(ids & gold)/max(1,len(gold))
    n=len(questions)
    return {"title_boost":boost,"n":n,
            **{f"complete_support_at_{k}":round(found[k]/n,6) for k in topk},
            **{f"support_recall_at_{k}":round(fraction[k]/n,6) for k in topk}}


def sweep(data:Path,output:Path,boosts:tuple[float,...]=(0.0,0.25,0.5,1.0,2.0,4.0)) ->dict:
    if output.exists(): raise FileExistsError(output)
    if 0.0 not in boosts:raise ValueError("must include unboosted baseline")
    results=[score_retriever(data,b) for b in boosts]
    # Select solely with PILOT. Primary full support top3; secondary avg recall.
    # Tie favors weaker inductive bias (lower coefficient).
    best=max(results,key=lambda r:(r["complete_support_at_3"],
                                   r["support_recall_at_3"],-r["title_boost"]))
    report={"partition":"pilot only, not heldout",
            "selection_rule":"max complete_support_at_3, then support_recall_at_3, then lower boost",
            "candidate_boosts":list(boosts),"results":results,
            "chosen_title_boost":best["title_boost"],
            "guard":"Do not re-tune after viewing heldout results"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(sweep(a.pilot,a.out),indent=2))


if __name__=="__main__":
    main()
