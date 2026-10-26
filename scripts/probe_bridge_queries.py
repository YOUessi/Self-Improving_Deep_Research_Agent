"""DEV-PILOT retrieval exposure diagnostic, not an agent answer benchmark.

For each pilot question:
  search(question) Top3 -> pretend to READ top1 retrieved document ->
  propose a second query from question and OBSERVED top1 title/body (no gold) ->
  search(second query) Top3 -> score union of both search results offline.

The runner's gold labels are used for aggregate recall *after* both tool
queries are selected, never by bridge_candidates. This is an upper-bound
diagnostic because the actual LLM may not read the first hit.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sidra.bridge import next_bridge_query
from sidra.io import load_dataset,save_jsonl
from sidra.retrieval import BM25Index


def probe(dataset:Path,out:Path, *,k:int=3,limit:int=0)->dict:
    if out.exists():raise FileExistsError(out)
    docs,qs=load_dataset(dataset)
    if any(q.split!="dev" for q in qs):
        raise ValueError("Pilot data must all be public DEV")
    index=BM25Index(docs)
    doc_lookup={d.doc_id:d for d in docs}
    cases=[]
    selected=sorted(qs,key=lambda q:q.question_id)
    if limit:selected=selected[:limit]
    for q in selected:
        first=index.search(q.question,scope=q.scope,k=k)
        first_ids={h.doc_id for h in first}
        gold=set(q.supporting_doc_ids)
        after=set(first_ids)
        query=None
        new_hits=[]
        if first:
            d=doc_lookup[first[0].doc_id]
            obs=[{"kind":"search","query":q.question,
                  "hits":[h.as_dict() for h in first]},
                 {"kind":"read","doc_id":d.doc_id,"title":d.title,
                  "text":d.text[:3500]}]
            candidate=next_bridge_query(q.question,obs)
            if candidate is not None:
                query=candidate.query
                new_hits=index.search(query,scope=q.scope,k=k)
                after.update(h.doc_id for h in new_hits)
        cases.append({
          "question_id":q.question_id,
          "bridge_query":query,
          "initial_hits":[h.doc_id for h in first],
          "added_hits":[h.doc_id for h in new_hits],
          "initial_support_complete":bool(gold) and gold.issubset(first_ids),
          "with_bridge_support_complete":bool(gold) and gold.issubset(after),
          "initial_support_recall":len(gold & first_ids)/max(1,len(gold)),
          "with_bridge_support_recall":len(gold & after)/max(1,len(gold)),
        })
    n=len(cases)
    if not n:raise ValueError("empty dataset")
    report={
      "data_partition":"PILOT DEV ONLY",
      "n":n,"top_k_each_query":k,"question_used_no_gold":True,
      "read_strategy":"top1 from initial search (gold-blind)",
      "initial_complete":sum(x["initial_support_complete"] for x in cases),
      "bridge_complete":sum(x["with_bridge_support_complete"] for x in cases),
      "bridge_queries":sum(bool(x["bridge_query"]) for x in cases),
      "initial_support_recall":round(sum(x["initial_support_recall"] for x in cases)/n,5),
      "bridge_support_recall":round(sum(x["with_bridge_support_recall"] for x in cases)/n,5),
      "warning":"NOT a full Agent result; two searches may incur higher tool cost; gold scored evaluator-only",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    save_jsonl(out/"pilot_diagnostics.jsonl",cases)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--k",type=int,default=3)
    a=p.parse_args()
    print(json.dumps(probe(a.pilot,a.out,k=a.k),indent=2))


if __name__=="__main__":
    main()
