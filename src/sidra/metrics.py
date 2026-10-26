"""Strict QA / evidence evaluation. Answer labels stay outside policy input."""

from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Any

from sidra.schema import AgentTrace, Document, Question

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    return 2 * common / (len(pred) + len(ref)) if common else 0.0


def score_trace(question: Question, trace: AgentTrace, docs: dict[str, Document]) -> dict[str, Any]:
    if question.question_id != trace.question_id:
        raise ValueError("Mismatch between trace and question ID")
    gold = set(question.supporting_doc_ids)
    cited = set(trace.citations)
    evidence_precision = len(cited & gold) / len(cited) if cited else 0.0
    evidence_recall = len(cited & gold) / len(gold) if gold else None
    texts = " ".join(docs[c].text for c in cited if c in docs)
    # This check is intentionally conservative; synonyms can be grounded yet fail substring.
    answer_in_evidence = bool(trace.answer.strip()) and normalize_answer(trace.answer) in normalize_answer(texts)
    exact_match = int(bool(trace.answer.strip()) and normalize_answer(trace.answer) == normalize_answer(question.gold_answer))
    grounded = bool(cited) and len(cited) == len(trace.citations) and answer_in_evidence
    evidence_complete = evidence_recall is None or evidence_recall == 1.0
    return {
        "question_id": question.question_id,
        "split": question.split,
        "policy": trace.policy,
        "exact_match": exact_match,
        "token_f1": round(token_f1(trace.answer, question.gold_answer), 6),
        "evidence_precision": round(evidence_precision, 6),
        "evidence_recall": round(evidence_recall, 6) if evidence_recall is not None else None,
        "answer_in_cited_evidence": bool(answer_in_evidence),
        "verified_success": int(bool(exact_match and grounded and evidence_complete)),
        "tool_calls": trace.tool_calls,
        "search_calls": trace.search_calls,
        "read_calls": trace.read_calls,
        "elapsed_ms": trace.elapsed_ms,
        "model_calls": trace.model_calls,
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "termination": trace.termination,
        "abstained": int(not trace.answer.strip()),
    }


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("Cannot summarize zero examples")
    fields = ["exact_match", "token_f1", "evidence_precision", "verified_success",
              "tool_calls", "search_calls", "read_calls", "elapsed_ms", "abstained",
              "model_calls", "prompt_tokens", "completion_tokens"]
    result = {"n": len(metrics), "policy": metrics[0]["policy"], "split": metrics[0]["split"]}
    for field in fields:
        result[field + "_mean"] = round(mean(m[field] for m in metrics), 6)
    valid_recall = [m["evidence_recall"] for m in metrics if m["evidence_recall"] is not None]
    result["evidence_recall_mean"] = round(mean(valid_recall), 6) if valid_recall else None
    result["terminations"] = dict(Counter(m["termination"] for m in metrics))
    return result
