"""Offline HotpotQA sentence-level evidence exposure audit.

The original HotpotQA labels have (title, sentence_index), unlike SIDRA's
passage-level supporting_doc_ids. This script checks whether all original
labelled SUPPORTING SENTENCES actually appeared in READ observations and
whether their passages were cited, without showing any gold fact to the Agent.

This is post-hoc EVALUATION ONLY, not a verifier reachable from the model.
Even exact labelled sentence exposure is *not* proof of semantic entailment.
For heldout dev, emit only aggregate counts (no per-QA gold-derived rows).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sidra.io import iter_jsonl
from sidra.metrics import normalize_answer


def norm(text:str)->str:
    return re.sub(r"\s+"," ",text.casefold()).strip()


def audit_case(original:dict,trace:dict)->dict:
    if original["_id"]!=trace["question_id"]:
        raise ValueError("question id mismatch")
    contexts=dict(original["context"])
    gold_pairs=[(str(title),int(idx)) for title,idx in original["supporting_facts"]]
    if not gold_pairs:
        raise ValueError("no labelled support sentences")
    observed_reads={}
    for step in trace["steps"]:
        obs=step.get("observation") or {}
        if obs.get("kind")=="read" and obs.get("doc_id"):
            observed_reads[str(obs["doc_id"])]={
                "title":str(obs.get("title","")),
                "text":str(obs.get("text",""))}
    cited=set(str(x) for x in trace.get("citations",[]))
    all_gold_titles={title for title,_ in gold_pairs}
    good_citations={doc_id for doc_id,read in observed_reads.items()
                    if doc_id in cited and read["title"] in all_gold_titles}
    sentence_read=0
    sentence_cited=0
    missing_out_of_range=0
    for title,index in gold_pairs:
        original_sentences=contexts.get(title)
        if original_sentences is None or index<0 or index>=len(original_sentences):
            missing_out_of_range+=1
            continue
        target=norm(original_sentences[index])
        # Mark empty or whitespace-only support sentences as unverified.
        if not target:
            continue
        in_read=False
        in_cited=False
        for doc_id,obs in observed_reads.items():
            if obs["title"]!=title:continue
            found=target in norm(obs["text"])
            if found:in_read=True
            if found and doc_id in cited:in_cited=True
        sentence_read+=int(in_read)
        sentence_cited+=int(in_cited)
    answer_exact=int(normalize_answer(trace.get("answer","")) ==
                     normalize_answer(original["answer"]))
    gold_passages_cited={observed_reads[i]["title"] for i in good_citations}
    complete_passage=all_gold_titles.issubset(gold_passages_cited)
    all_sentences_read=sentence_read==len(gold_pairs)
    all_sentences_cited=sentence_cited==len(gold_pairs)
    return {
        "answer_exact":answer_exact,
        "original_support_facts_count":len(gold_pairs),
        "supporting_sentences_read":sentence_read,
        "supporting_sentences_cited":sentence_cited,
        "passage_support_complete":bool(complete_passage),
        "sentence_support_read_complete":all_sentences_read,
        "sentence_support_cited_complete":all_sentences_cited,
        "answer_and_sentence_support_complete":bool(
            answer_exact and all_sentences_cited and complete_passage),
        "invalid_original_sentence_indices":missing_out_of_range,
        "citation_precision":len(good_citations)/len(cited) if cited else 0.0,
    }


def audit(original:Path,traces:Path,out:Path,*,heldout:bool=True)->dict:
    if out.exists():raise FileExistsError(out)
    questions=json.loads(original.read_text())
    byid={row["_id"]:row for row in questions}
    if len(byid)!=len(questions):raise ValueError("duplicate original question IDs")
    traces_list=list(iter_jsonl(traces))
    ids=[x["question_id"] for x in traces_list]
    if not ids or len(ids)!=len(set(ids)) or set(ids)!=set(byid):
        raise ValueError("must evaluate exactly this full frozen question set")
    rows=[audit_case(byid[t["question_id"]],t) for t in traces_list]
    n=len(rows)
    fields=[
        "answer_exact","passage_support_complete",
        "sentence_support_read_complete","sentence_support_cited_complete",
        "answer_and_sentence_support_complete",
    ]
    report={
        "dataset":"original HotpotQA distractor DEV, evaluator-only",
        "n":n,"label_granularity":"original (title,sentence_index)",
        "heldout_aggregate_only":heldout,
        **{key+"_count":sum(int(x[key]) for x in rows) for key in fields},
        "mean_supporting_sentence_read_fraction":round(
            sum(x["supporting_sentences_read"]/x["original_support_facts_count"]
                for x in rows)/n,6),
        "mean_supporting_sentence_cited_fraction":round(
            sum(x["supporting_sentences_cited"]/x["original_support_facts_count"]
                for x in rows)/n,6),
        "invalid_source_fact_indices":sum(x["invalid_original_sentence_indices"]
                                           for x in rows),
        "meaning":"sentence presence in cited READ text; not entailment or independent web citation verification",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    if not heldout:
        from sidra.io import save_jsonl
        save_jsonl(out/"pilot_per_case.jsonl",[
            {"question_id":t["question_id"],**row}
            for t,row in zip(traces_list,rows)])
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--traces",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--pilot",action="store_true",
                   help="Permit per-example report only for public DEV pilot")
    a=p.parse_args()
    print(json.dumps(audit(a.original,a.traces,a.out,heldout=not a.pilot),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
