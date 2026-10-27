"""Post-hoc Qwen3-8B answer + TRAIN-only ranker, using fixed ACTUAL READs.

No new 8B calls, no new Search/Read and no TRAIN/DEV gold enters the citation
ranker. The evaluator checks true pilot labels ONLY after frozen predictions
are composed. Compare with original, unmodified fixed-8B solver predictions.
This is a module-level pilot, not an end-to-end production Agent benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from sidra.evidence_ranker import EvidenceRanker
from sidra.io import load_dataset,iter_jsonl,save_jsonl
from sidra.metrics import score_trace
from sidra.schema import AgentTrace
from scripts.m7_fixed_evidence_8b_solver_pilot import read_sources,sha256_file


def compare(pilot:Path,traces:Path,previous_solver:Path,
            ranker_path:Path,out:Path)->dict:
    if out.exists():raise FileExistsError(out)
    if pilot.name!="converted" or pilot.parent.name!="pilot":
        raise ValueError("Only public DEV pilot32 allowed; final-B160 sealed")
    docs,qs=load_dataset(pilot)
    if len(qs)!=32 or any(q.split!="dev" for q in qs):
        raise ValueError("expected frozen original DEV pilot32")
    bydoc={d.doc_id:d for d in docs}
    byq={q.question_id:q for q in qs}
    model=EvidenceRanker.from_file(ranker_path)
    original={t["question_id"]:AgentTrace(**t) for t in iter_jsonl(traces)}
    prior={r["question_id"]:r for r in iter_jsonl(previous_solver)}
    if len(original)!=32 or len(prior)!=32 or set(original)!=set(prior)!=set(byq):
        raise ValueError("pilot trace / 8B answer prediction IDs do not match")
    n_answer=0
    baseline_score=0
    raw_correct=0
    corrected_correct=0
    corrected_verified=0
    filled=0
    ranker_abstentions=0
    predicted_gold_support=0
    supported_read=0
    rows=[]
    for qid in sorted(byq):
        q=byq[qid]
        trace=original[qid]
        if trace.question!=q.question:
            raise ValueError("question body differs from immutable source")
        before=score_trace(q,trace,bydoc)
        baseline_score+=before["verified_success"]
        r=prior[qid]
        if not r.get("solver_prediction"):
            answer=""
        else:
            answer=str(r["solver_prediction"].get("answer",""))
        reads=read_sources(trace)
        n_answer+=int(bool(answer))
        pred=model.select(q.question,answer,reads)
        answer_scored=score_trace(q,replace(trace,answer=answer,
                             citations=[],policy="8b_raw_answer_no_cite"),bydoc)
        raw_correct+=answer_scored["exact_match"]
        corrected=score_trace(q,replace(trace,answer=pred["answer"],
                          citations=pred["citations"],
                          policy="offline_train_ranker_citations"),bydoc)
        corrected_correct+=corrected["exact_match"]
        corrected_verified+=corrected["verified_success"]
        filled+=int(bool(pred["citations"]))
        ranker_abstentions+=int(not pred["answer"])
        gold=set(q.supporting_doc_ids)
        have=set(x["doc_id"] for x in reads)
        supported_read+=int(gold.issubset(have))
        predicted_gold_support+=int(bool(pred["citations"]) and
                                    gold.issubset(set(pred["citations"])))
        rows.append({
            "question_id":qid,
            "n_read":len(reads),
            "raw_8b_answer_exact":answer_scored["exact_match"],
            "train_ranker_gated_exact":corrected["exact_match"],
            "train_ranker_strict_verified":corrected["verified_success"],
            "all_gold_supports_read":gold.issubset(have),
            "all_gold_supports_selected":bool(pred["citations"]) and
                                          gold.issubset(set(pred["citations"])),
            "citation_selector_reason":pred["reason"],
            "attributed_doc_ids":pred["citations"],
            "source_quotes":pred["evidence"],
        })
    report={
        "stage":"M7_TRAIN_ONLY_LEARNED_EVIDENCE_RANKER_ON_FIXED_8B_ANSWER_PILOT",
        "n":32,
        "source":"already-executed actual Qwen3-0.6B Search/Read + stored 8B answer",
        "source_trace_sha256":sha256_file(traces),
        "existing_8b_prediction_file_sha256":sha256_file(previous_solver),
        "ranker_model_sha256":sha256_file(ranker_path),
        "ranker_training_only_original_train":True,
        "online_solver_received_DEV_gold":False,
        "evaluator_only_gold":True,
        "fixed_8b_answers_available":n_answer,
        "raw_8b_answer_exact_count":raw_correct,
        "ranker_citation_attributed_count":filled,
        "ranker_abstention_count":ranker_abstentions,
        "ranker_gated_answer_exact_count":corrected_correct,
        "ranker_strict_verified_count":corrected_verified,
        "gold_pair_actually_read_count":supported_read,
        "gold_pair_selected_count":predicted_gold_support,
        "extra_llm_calls":0,
        "actual_tool_calls_changed":False,
        "training_or_LLM_finetuning_in_this_step":False,
        "semantic_entailment_verified":False,
        "scope_warning":"DEV pilot32 posthoc module study, not re-run Agent, not unseen hidden test",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    save_jsonl(out/"pilot_attribution.jsonl",rows)
    return report


def main():
    a=argparse.ArgumentParser()
    for arg in ("pilot","traces","previous-solver","ranker","out"):
        a.add_argument("--"+arg,required=True,type=Path)
    cfg=a.parse_args()
    print(json.dumps(compare(cfg.pilot,cfg.traces,cfg.previous_solver,
                             cfg.ranker,cfg.out),indent=2))


if __name__=="__main__":
    main()
