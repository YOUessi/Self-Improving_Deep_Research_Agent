"""Offline trace-forensics / evidence-gate attribution on public labeled dev.

Gold labels are read only by this *post hoc evaluator*, never by an online
policy or search tool. Error categories are diagnostics, not a new reward.

Examples:
  python scripts/diagnose_traces.py --dataset <pilot/converted> \
    --traces <run/traces.jsonl> --out <new-dir> --mode pilot
  python scripts/diagnose_traces.py --dataset <heldout/converted> \
    --traces <run/traces.jsonl> --out <new-dir> --mode heldout
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from sidra.io import iter_jsonl, load_dataset, save_jsonl
from sidra.metrics import normalize_answer, score_trace
from sidra.schema import AgentTrace, Document, Question


def classify(question: Question, trace: AgentTrace,
             doc_lookup: dict[str, Document]) -> dict:
    """Separate search exposure, reading, citation, answer and verifier failures."""
    score = score_trace(question, trace, doc_lookup)
    gold = set(question.supporting_doc_ids)
    seen: set[str] = set()
    read: set[str] = set()
    searches: list[str] = []
    reads: list[str] = []
    verifier_reasons: list[str] = []
    for step in trace.steps:
        act = step.get("action") or {}
        obs = step.get("observation") or {}
        if act.get("kind") == "search":
            searches.append(str(act.get("query", "")))
            seen.update(str(hit["doc_id"]) for hit in obs.get("hits", []))
        if act.get("kind") == "read":
            reads.append(str(act.get("doc_id", "")))
            if obs.get("kind") == "read" and obs.get("doc_id"):
                read.add(str(obs["doc_id"]))
        audit = step.get("verification") or {}
        if not audit.get("approved", True):
            verifier_reasons.append(str(audit.get("reason", "unspecified")))
    cited = set(trace.citations)
    search_support = len(gold & seen)
    read_support = len(gold & read)
    cite_support = len(gold & cited)
    reason = ""
    if trace.termination == "error":
        err = trace.error.lower()
        reason = ("malformed_action_json" if ("agent action json" in err or
                                             "invalid action kind" in err) else
                  "illegal_tool_or_citation" if ("read requires" in err or
                                                 "unverified citation" in err) else
                  "runtime_error")
    elif score["verified_success"]:
        reason = "strict_success"
    elif not score["exact_match"] and trace.answer:
        reason = "wrong_answer"
    elif not trace.answer and verifier_reasons:
        reason = "verifier_rejection_or_abstention"
    elif not trace.answer:
        reason = "no_answer_or_budget"
    elif search_support < len(gold):
        reason = "not_all_supports_retrieved"
    elif read_support < len(gold):
        reason = "supports_retrieved_not_read"
    elif cite_support < len(gold):
        reason = "supports_read_not_cited"
    elif not score["answer_in_cited_evidence"]:
        reason = "literal_answer_not_in_citations"
    else:
        reason = "other_nonverified"

    # Diagnostic bottleneck can coexist with wrong answer: do not let the
    # first reason obscure retrieval and citation failures.
    if len(gold) == 0:
        bottleneck = "unlabelled_evidence"
    elif search_support < len(gold):
        bottleneck = "retrieval_search_exposure"
    elif read_support < len(gold):
        bottleneck = "tool_read_selection"
    elif cite_support < len(gold):
        bottleneck = "citation_selection"
    elif not score["exact_match"]:
        bottleneck = "answer_reasoning"
    elif not score["answer_in_cited_evidence"]:
        bottleneck = "lexical_verifier_overreject_possible"
    else:
        bottleneck = "all_gates_passed_or_other"
    return {
        **score,
        "reason": reason, "bottleneck": bottleneck,
        "support_gold_count": len(gold),
        "support_exposed_count": search_support,
        "support_read_count": read_support,
        "support_cited_count": cite_support,
        "search_support_recall": search_support / len(gold) if gold else None,
        "read_support_recall": read_support / len(gold) if gold else None,
        "read_before_cite_ok": cited.issubset(read),
        "exposed_both_supports": bool(gold) and gold.issubset(seen),
        "read_both_supports": bool(gold) and gold.issubset(read),
        "cited_both_supports": bool(gold) and gold.issubset(cited),
        "repeated_read_count": len(reads) - len(set(reads)),
        "repeated_search_count": len(searches) - len(set(x.casefold() for x in searches)),
        "gold_answer_is_yes_no": normalize_answer(question.gold_answer) in {"yes", "no"},
        "verifier_reasons": verifier_reasons,
        "error_type": trace.error.split(":", 1)[0] if trace.error else None,
    }


def _wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0, 0]
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [round(max(0, center-margin), 4), round(min(1, center+margin), 4)]


def audit_run(dataset: Path, traces: Path, output: Path, mode: str) -> dict:
    if mode not in {"pilot", "heldout"}:
        raise ValueError("mode must be pilot or heldout")
    if output.exists():
        raise FileExistsError(output)
    docs, questions = load_dataset(dataset)
    qmap = {q.question_id:q for q in questions}
    dmap = {d.doc_id:d for d in docs}
    trace_rows = list(iter_jsonl(traces))
    qids = [row["question_id"] for row in trace_rows]
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate trace question IDs")
    if not qids or set(qids) - set(qmap):
        raise ValueError("empty run or trace not in dataset")
    if any(qmap[qid].split != "dev" for qid in qids):
        raise ValueError("expects frozen dev; train/test traces are not accepted")
    classified = [classify(qmap[row["question_id"]], AgentTrace(**row), dmap)
                  for row in trace_rows]
    n = len(classified)
    counts = lambda field: dict(sorted(Counter(row[field] for row in classified).items()))
    sum_field = lambda field: sum(int(bool(row[field])) for row in classified)
    success = sum_field("verified_success")
    exact = sum_field("exact_match")
    summary = {
        "mode": mode, "n": n,
        "answer_exact": exact, "answer_exact_rate": round(exact/n, 5),
        "strict_verified": success, "strict_verified_rate": round(success/n, 5),
        "exact_ci95_wilson": _wilson(exact, n),
        "verified_ci95_wilson": _wilson(success, n),
        "support_exposed_full": sum_field("exposed_both_supports"),
        "support_read_full": sum_field("read_both_supports"),
        "support_cited_full": sum_field("cited_both_supports"),
        "repeated_read_cases": sum(int(row["repeated_read_count"] > 0)
                                   for row in classified),
        "repeated_search_cases": sum(int(row["repeated_search_count"] > 0)
                                     for row in classified),
        "no_final_answer": sum(int(row["abstained"]) for row in classified),
        "termination_counts": counts("termination"),
        "reason_counts": counts("reason"),
        "bottleneck_counts": counts("bottleneck"),
        "verifier_reasons": dict(sorted(Counter(
            reason for row in classified for reason in row["verifier_reasons"]).items())),
        "gold_yes_no_cases": sum_field("gold_answer_is_yes_no"),
        "gold_citations_correct_but_literal_rejected": sum(
            int(row["exact_match"] and row["cited_both_supports"] and
                not row["answer_in_cited_evidence"]) for row in classified),
        "label_usage": "EVALUATOR-ONLY; never feed gold support, answers or these categories to Agent",
        "limitation": "support evidence measured at passage TITLE-level, not original sentence-level",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    if mode == "pilot":
        save_jsonl(output/"per_example.jsonl", classified)
    # No per-item gold-derived diagnostics are exported for heldout.
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,type=Path)
    ap.add_argument("--traces",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--mode",required=True,choices=("pilot","heldout"))
    args=ap.parse_args()
    print(json.dumps(audit_run(args.dataset,args.traces,args.out,args.mode),indent=2))


if __name__=="__main__":
    main()
