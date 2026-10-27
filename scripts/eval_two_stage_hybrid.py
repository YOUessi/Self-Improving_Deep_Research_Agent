"""Two-phase real SearchAgent + Qwen3-8B answer + TRAIN-fitted evidence selector.

Stage 1 is a genuine bounded 0.6B SearchAgent episode, executed and frozen
BEFORE stage 2. Stage 2 calls Qwen3-8B-AWQ on ONLY the genuine READ
observations; it runs a fixed evidence ranker trained on a disjoint TRAIN1024.
No original TRAIN64 answer/support labels are passed to either model.

This is a genuine *two-stage modular system with stored intermediate tool
trajectories*, not a single synchronous Agent runtime and NOT an RSI update.
Gold 64-question answers/support labels are evaluator-only AFTER all 8B outputs
have been produced. Source quotation matching is not semantic entailment.

Inputs require an immutable, preregistered, previously unexposed original
TRAIN64 cohort. Never accept DEV/final-B160 or original training1024 here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import random
from pathlib import Path

from sidra.evidence_ranker import EvidenceRanker
from sidra.io import load_dataset,iter_jsonl,save_jsonl
from sidra.metrics import score_trace
from sidra.schema import AgentTrace
from scripts.audit_hotpot_sentence_evidence import audit_case
from scripts.m7_fixed_evidence_8b_solver_pilot import (
    get_solver_output,read_sources,solver_request,sha256_file
)


def _bootstrap_delta(values:list[float],*,seed:int=712,draws:int=5000)->list[float]:
    if len(values)<32:raise ValueError("at least 32 independent question clusters required")
    rng=random.Random(seed)
    n=len(values)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                   for __ in range(draws))
    return [round(samples[int((draws-1)*0.025)],6),
            round(samples[int((draws-1)*0.975)],6)]


def compose(question:str,trace:AgentTrace,ranker:EvidenceRanker,
            *,endpoint:str,model:str)->dict:
    """No gold is accepted by signature. Extra 8B cost is mandatory."""
    reads=read_sources(trace)
    if len(reads)<2:
        return {"solver_answer":"","solver_citations":[],
                "answer":"","citations":[],"evidence":[],
                "reason":"under_two_real_reads",
                "llm_calls":0,"prompt_tokens":0,"completion_tokens":0,
                "model_elapsed_ms":0}
    try:
        raw,usage,seconds=get_solver_output(
            endpoint,solver_request(question,reads,model=model),timeout=75)
        ans=str(raw["answer"]).strip()
        selected=ranker.select(question,ans,reads)
        returned={
            "solver_answer":ans,
            "solver_citations":raw["citations"],
            "answer":selected["answer"],
            "citations":selected["citations"],
            "evidence":selected["evidence"],
            "reason":selected["reason"],
            "llm_calls":1,
            "prompt_tokens":int(usage.get("prompt_tokens") or 0),
            "completion_tokens":int(usage.get("completion_tokens") or 0),
            "model_elapsed_ms":int(seconds*1000),
        }
        if any(
            not any(e["quote"] in r["text"] for r in reads
                    if r["doc_id"]==e["doc_id"])
            for e in returned["evidence"]
        ):
            raise ValueError("ranker output contains non-literal READ quotation")
        return returned
    except (OSError,ValueError,KeyError,TypeError,TimeoutError) as exc:
        return {"solver_answer":"","solver_citations":[],
                "answer":"","citations":[],"evidence":[],
                "reason":"solver_or_ranker_error_"+type(exc).__name__,
                "llm_calls":1,"prompt_tokens":0,"completion_tokens":0,
                "model_elapsed_ms":0}


def evaluate(frozen_root:Path,base_run:Path,ranker_path:Path,out:Path,
             *,endpoint:str,model:str,expected_ranker_sha:str)->dict:
    if out.exists():raise FileExistsError(out)
    if frozen_root.name!="frozen_unseen_train64_s71212026":
        raise ValueError("requires preregistered new TRAIN64, not DEV/other")
    manifest=json.loads((frozen_root/"FREEZE_MANIFEST.json").read_text())
    if (manifest.get("stage")!="M7_NEW_QUESTION_TRAIN64_EVAL_ONLY"
        or manifest.get("optimizer_train_eligible") is not False
        or manifest.get("n")!=64 or not manifest.get("question_id_exclusion_verified")):
        raise ValueError("invalid new evaluation-only cohort manifest")
    source=frozen_root/"converted"
    if (sha256_file(source/"questions.jsonl")!=manifest["converted_questions_sha256"]
        or sha256_file(source/"documents.jsonl")!=manifest["converted_documents_sha256"]):
        raise ValueError("original heldout TRAIN64 data modified after freezing")
    if sha256_file(ranker_path)!=expected_ranker_sha:
        raise ValueError("trained citation ranker SHA changed after preregistration")
    ranker=EvidenceRanker.from_file(ranker_path)
    if ranker.train_data_sha256!=manifest["prior_train1024_questions_sha256"]:
        raise ValueError("ranker was not trained on the strictly excluded original TRAIN1024")
    baseline_manifest=json.loads((base_run/"manifest.json").read_text())
    baseline_policy=json.loads((base_run/"FINITE_ACTION_POLICY.json").read_text())
    if (baseline_manifest.get("split")!="train"
        or baseline_manifest.get("sample_count")!=64
        or baseline_manifest["data_sha256"]!={"documents.jsonl":manifest["converted_documents_sha256"],
                                             "questions.jsonl":manifest["converted_questions_sha256"]}
        or baseline_policy.get("answer_proposer")!="v4"
        or baseline_policy.get("selection_mode")!="sample"
        or baseline_policy.get("trained_on_eval_gold") is not False):
        raise ValueError("frozen V4 SearchAgent baseline provenance invalid")
    docs,questions=load_dataset(source)
    if len(questions)!=64 or any(q.split!="train" for q in questions):
        raise ValueError("TRAIN64 evaluator membership/split mismatch")
    qids={q.question_id for q in questions}
    traces=list(iter_jsonl(base_run/"traces.jsonl"))
    if (len(traces)!=64 or len({t["question_id"] for t in traces})!=64
        or {t["question_id"] for t in traces}!=qids):
        raise ValueError("baseline tool traces and TRAIN64 membership mismatch")
    original_file=frozen_root/"original_eval_only_train_subset.json"
    if sha256_file(original_file)!=manifest["original_subset_sha256"]:
        raise ValueError("original sentence-level source bytes changed")
    original_by_id={r["_id"]:r for r in json.loads(original_file.read_text())}
    doc_map={d.doc_id:d for d in docs}
    qmap={q.question_id:q for q in questions}

    # Phase A: generate ALL solver outputs without consulting any q.gold_answer
    # or supporting_doc_ids. Gold enters only the independent scoring pass.
    generated={}
    logs=[]
    for i,row in enumerate(traces):
        t=AgentTrace(**row)
        if qmap[t.question_id].question!=t.question:
            raise ValueError("baseline question text mismatch")
        pred=compose(t.question,t,ranker,endpoint=endpoint,model=model)
        generated[t.question_id]=pred
        logs.append({
            "question_id":t.question_id,"question":t.question,
            "input_read_document_ids":[r["doc_id"] for r in read_sources(t)],
            "solver_answer":pred["solver_answer"],
            "ranker_answer":pred["answer"],
            "ranker_citations":pred["citations"],
            "literal_READ_evidence":pred["evidence"],
            "reason":pred["reason"],
            "llm_calls":pred["llm_calls"],
            "prompt_tokens":pred["prompt_tokens"],
            "completion_tokens":pred["completion_tokens"],
            "model_elapsed_ms":pred["model_elapsed_ms"],
            "no_evaluator_labels_in_model_call":True,
        })
        if (i+1)%8==0:
            print("HYBRID_GENERATED",i+1,"of",len(traces),flush=True)

    # Phase B: offline evaluator ONLY AFTER all model calls completed.
    measures=("exact_match","verified_success","evidence_recall")
    base_scores=[]
    solver_scores=[]
    hybrid_scores=[]
    original_sentence_scores=[]
    metrics={"llm_calls":0,"solver_prompt_tokens":0,"solver_completion_tokens":0,
             "solver_elapsed_ms":0,"ranker_abstain_count":0,"solver_errors":0,
             "sufficient_READ_count":0}
    corrected_traces=[]
    for row in traces:
        t=AgentTrace(**row)
        q=qmap[t.question_id]
        prediction=generated[t.question_id]
        base=score_trace(q,t,doc_map)
        raw=replace(t,answer=prediction["solver_answer"],
                    citations=list(prediction["solver_citations"]),
                    policy="Qwen3-8B_raw_second_stage_answer")
        raw_score=score_trace(q,raw,doc_map)
        corrected=replace(
            t,answer=prediction["answer"],
            citations=list(prediction["citations"]),
            policy="0.6B_SearchRead__8B_answer__TRAIN_ranker",
            termination="finish" if prediction["answer"] else "abstained",
            model_calls=t.model_calls+prediction["llm_calls"],
            prompt_tokens=t.prompt_tokens+prediction["prompt_tokens"],
            completion_tokens=t.completion_tokens+prediction["completion_tokens"],
            elapsed_ms=t.elapsed_ms+prediction["model_elapsed_ms"],
        )
        model_score=score_trace(q,corrected,doc_map)
        base_scores.append(base)
        solver_scores.append(raw_score)
        hybrid_scores.append(model_score)
        source_result=audit_case(original_by_id[t.question_id],corrected.as_dict())
        original_sentence_scores.append(source_result)
        corrected_traces.append(corrected.as_dict())
        metrics["llm_calls"]+=prediction["llm_calls"]
        metrics["solver_prompt_tokens"]+=prediction["prompt_tokens"]
        metrics["solver_completion_tokens"]+=prediction["completion_tokens"]
        metrics["solver_elapsed_ms"]+=prediction["model_elapsed_ms"]
        metrics["ranker_abstain_count"]+=int(not prediction["answer"])
        metrics["solver_errors"]+=int(prediction["reason"].startswith("solver_or_ranker_error_"))
        metrics["sufficient_READ_count"]+=int(len(read_sources(t))>=2)
    base_correct=sum(x["exact_match"] for x in base_scores)
    raw_correct=sum(x["exact_match"] for x in solver_scores)
    hybrid_correct=sum(x["exact_match"] for x in hybrid_scores)
    base_verified=sum(x["verified_success"] for x in base_scores)
    hybrid_verified=sum(x["verified_success"] for x in hybrid_scores)
    base_recall=sum(x["evidence_recall"] for x in base_scores)/64
    hybrid_recall=sum(x["evidence_recall"] for x in hybrid_scores)/64
    base_tool=sum(t["tool_calls"] for t in traces)/64
    base_cost=sum(t["model_calls"] for t in traces)/64
    base_latency=sum(t["elapsed_ms"] for t in traces)/64
    result={
        "stage":"M7_FRESH_TRAIN64_REAL_0.6B_SEARCH_READ_THEN_8B_ANSWER_TRAIN_EVIDENCE",
        "n_unique_original_TRAIN_question_ids":64,
        "source_train_cohort_qid_sha256":manifest["ordered_question_ids_sha256"],
        "frozen_cohort_sha256":sha256_file(frozen_root/"FREEZE_MANIFEST.json"),
        "baseline_git_commit":baseline_manifest["git_commit"],
        "baseline_policy":"finite categorical V4, real SearchAgent tool paths",
        "base_adapter_sha256":baseline_policy["source_adapter_sha256"],
        "answer_model":model,
        "answer_ranker_sha256":expected_ranker_sha,
        "ranker_trained_exclusively_prior_train1024":True,
        "new_eval64_excluded_from_all_train_and_threshold_fitting":True,
        "baseline_answer_correct":base_correct,
        "raw_8b_answer_correct":raw_correct,
        "hybrid_answer_correct":hybrid_correct,
        "baseline_strict_verified":base_verified,
        "hybrid_strict_verified":hybrid_verified,
        "hybrid_original_sentence_evidence_complete":sum(
            bool(s["answer_and_sentence_support_complete"]) for s in original_sentence_scores),
        "baseline_gold_doc_recall_mean":round(base_recall,6),
        "hybrid_gold_doc_recall_mean":round(hybrid_recall,6),
        "extra":metrics,
        "baseline_mean_tool_calls":round(base_tool,6),
        "hybrid_mean_tool_calls":round(base_tool,6),
        "baseline_mean_model_calls":round(base_cost,6),
        "hybrid_mean_model_calls":round(base_cost+metrics["llm_calls"]/64,6),
        "baseline_mean_elapsed_ms":round(base_latency,6),
        "hybrid_mean_elapsed_ms":round(base_latency+metrics["solver_elapsed_ms"]/64,6),
        "paired_bootstrap_answer_delta95":_bootstrap_delta([
            b["exact_match"]-a["exact_match"]
            for a,b in zip(base_scores,hybrid_scores)],seed=171),
        "paired_bootstrap_strict_delta95":_bootstrap_delta([
            b["verified_success"]-a["verified_success"]
            for a,b in zip(base_scores,hybrid_scores)],seed=919),
        "onpolicy_model_weight_updates_this_run":0,
        "DEV_final_B160_read":False,
        "honest_scope":"New disjoint QUESTION cohort from same original HotpotQA TRAIN, NOT external hidden benchmark. Two separate GPU phases on the same real SearchAgent traces; not a single streaming Agent runtime or GRPO/RSI.",
        "evidence_limit":"Document-passage + original sentence presence checks, NOT semantic entailment.",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    save_jsonl(out/"solver_and_ranker_outputs.jsonl",logs)
    save_jsonl(out/"final_hybrid_traces.jsonl",corrected_traces)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("frozen-root","baseline","ranker","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--ranker-sha256",required=True)
    p.add_argument("--endpoint",default="http://127.0.0.1:18988/v1/chat/completions")
    p.add_argument("--model",default="sidra-qwen3-8b-awq")
    a=p.parse_args()
    evaluate(a.frozen_root,a.baseline,a.ranker,a.out,
             endpoint=a.endpoint,model=a.model,
             expected_ranker_sha=a.ranker_sha256)


if __name__=="__main__":
    main()
