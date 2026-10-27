"""M7: offline, train-only bottleneck attribution vs observable verifier decisions.

Two disjoint trust domains:
(1) source_only_verdict(trace) sees NO Question, supporting IDs, gold answer,
    original supporting sentence indices or corpus contents. It checks only
    model-produced answer/citations and *actual READ observations* and NEVER
    claims semantic entailment.
(2) offline_train_diagnosis(question, trace, original) is a post-episode TRAIN
    evaluator that measures exposure, READ, citation, original HotpotQA
    (title, sentence_index) preservation and answer EM.

Do not import any offline diagnosis function into a model policy. Original
HotpotQA annotations must NEVER influence online actions or reward on DEV.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sidra.metrics import normalize_answer
from sidra.schema import AgentTrace, Question


@dataclass(frozen=True)
class SourceVerdict:
    """Observation-only failure taxonomy. Not a semantic entailment verdict."""

    reason: str
    lexical_candidate: bool
    citations_read: bool
    distinct_read_citations: int
    concat_only_lexical_candidate: bool
    semantic_entailment: str = "not_evaluated"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_reason": self.reason,
            "lexical_candidate": int(self.lexical_candidate),
            "all_citations_actually_read": int(self.citations_read),
            "distinct_read_citations": self.distinct_read_citations,
            "concat_only_lexical_candidate": int(self.concat_only_lexical_candidate),
            "semantic_entailment": self.semantic_entailment,
        }


def _present_in_one_document(answer: str, text: str) -> bool:
    """Normalized contiguous *word* sequence, not cross-document substrings."""
    a = normalize_answer(answer).split()
    s = normalize_answer(text).split()
    return bool(a) and any(s[i:i + len(a)] == a
                           for i in range(len(s) - len(a) + 1))


def _reads(trace: dict) -> dict[str, str]:
    """Only READ observations emitted by the actual SearchAgent are sources."""
    result: dict[str, str] = {}
    for step in trace.get("steps", []):
        o = step.get("observation") or {}
        a = step.get("action") or {}
        if (o.get("kind") == "read" and a.get("kind") == "read"
                and str(a.get("doc_id")) == str(o.get("doc_id"))
                and o.get("doc_id")):
            doc_id = str(o["doc_id"])
            result[doc_id] = str(o.get("text", ""))
    return result


def source_only_verdict(trace: dict[str, Any], *, min_citations: int = 2) -> SourceVerdict:
    """Strictly observation-only. 'lexical_only' is NEVER an NLI approval."""
    if min_citations < 1:
        raise ValueError("min_citations must be >=1")
    ans = str(trace.get("answer", ""))
    termination = str(trace.get("termination", ""))
    read = _reads(trace)
    cites = tuple(str(x) for x in trace.get("citations", []))
    unique = set(cites)
    valid_read_citations = unique.issubset(read)
    common = dict(
        lexical_candidate=False,
        citations_read=valid_read_citations,
        distinct_read_citations=len(unique & set(read)),
        concat_only_lexical_candidate=False,
    )

    if termination == "error":
        error = str(trace.get("error", "")).casefold()
        reason = ("unread_citation_action_error" if "unverified citations" in error
                  else "illegal_or_malformed_action")
        return SourceVerdict(reason, **common)
    if termination != "finish":
        reason = "abstained" if (termination == "abstained" or not ans.strip()) else "nonfinish_termination"
        return SourceVerdict(reason, **common)
    if not normalize_answer(ans):
        return SourceVerdict("empty_answer", **common)
    if len(unique) != len(cites):
        return SourceVerdict("duplicate_citation", **common)
    if len(unique) < min_citations:
        return SourceVerdict("too_few_citations", **common)
    if not valid_read_citations:
        return SourceVerdict("citation_not_read", **common)

    per_doc = any(_present_in_one_document(ans, read[doc]) for doc in unique)
    # For diagnostics only: old verifier's concatenation can create a
    # phantom match across a boundary between two different READ documents.
    concatenated = _present_in_one_document(
        ans, " ".join(read[doc] for doc in cites)
    )
    if not per_doc:
        return SourceVerdict(
            "lexical_answer_not_in_any_cited_read",
            lexical_candidate=False,
            citations_read=True,
            distinct_read_citations=len(unique),
            concat_only_lexical_candidate=concatenated,
        )
    return SourceVerdict(
        "lexical_only_entailment_unverified",
        lexical_candidate=True,
        citations_read=True,
        distinct_read_citations=len(unique),
        concat_only_lexical_candidate=False,
    )


def _exposed_read_cited(trace: dict) -> tuple[set[str], set[str], set[str]]:
    exposed: set[str] = set()
    for step in trace.get("steps", []):
        o = step.get("observation") or {}
        if o.get("kind") != "search" or (step.get("action") or {}).get("kind") != "search":
            continue
        exposed.update(str(h["doc_id"]) for h in (o.get("hits") or [])
                       if h.get("doc_id"))
    return exposed, set(_reads(trace)), set(map(str, trace.get("citations", [])))


def offline_train_diagnosis(
    question: Question, trace: dict[str, Any], original: dict[str, Any],
    *, min_citations: int = 2,
) -> dict[str, Any]:
    """Post-episode OFFLINE TRAIN evaluation with genuine original sentence facts.

    Returns a single earliest-missing-evidence bottleneck plus multiple
    independent success/quality metrics; hierarchy is attribution heuristic,
    NOT causal proof or a model-accessible reward function.
    """
    if question.split != "train":
        raise ValueError("TRAIN-only original sentence diagnosis refuses DEV/TEST")
    if not (question.question_id == trace.get("question_id") == original.get("_id")):
        raise ValueError("TRAIN original/converted/trace question IDs differ")
    if question.question != trace.get("question"):
        raise ValueError("trace question differs from frozen TRAIN question")
    # Import-only evaluator: never pass these gold labels into the policy.
    from scripts.audit_hotpot_sentence_evidence import audit_case

    sentence = audit_case(original, trace)
    if normalize_answer(question.gold_answer) != normalize_answer(str(original.get("answer", ""))):
        raise ValueError("original/converted gold answer mismatch")
    facts = original.get("supporting_facts", [])
    contexts = dict(original.get("context", []))
    if not facts or any(
        title not in contexts or not isinstance(i, int) or i < 0 or i >= len(contexts[title])
        for title, i in facts
    ):
        raise ValueError("invalid TRAIN original supporting sentence indices")

    gold = set(question.supporting_doc_ids)
    if not gold:
        raise ValueError("TRAIN support passage IDs required")
    exposed, read, cited = _exposed_read_cited(trace)
    src = source_only_verdict(trace, min_citations=min_citations)
    exposed_complete = gold <= exposed
    read_complete = gold <= read
    cited_complete = gold <= cited
    sentence_read = bool(sentence["sentence_support_read_complete"])
    sentence_cited = bool(sentence["sentence_support_cited_complete"])
    em = bool(sentence["answer_exact"])

    if not exposed_complete:
        stage = "support_not_retrieved"
    elif not read_complete:
        stage = "support_retrieved_not_read"
    elif not sentence_read:
        stage = "support_read_but_gold_sentence_unseen"
    elif not cited_complete:
        stage = "support_read_not_cited"
    elif not sentence_cited:
        stage = "support_sentence_not_in_cited_read"
    elif not em:
        stage = "answer_wrong_or_absent"
    elif not src.lexical_candidate:
        stage = "answer_lexically_unanchored"
    else:
        stage = "answer_and_sentence_exposed_semantics_unverified"

    return {
        "question_id": question.question_id,
        "rollout_index": trace.get("rollout_index"),
        "primary_evidence_bottleneck": stage,
        "source_reason": src.reason,
        "answer_em": int(em),
        "support_retrieval_complete": int(exposed_complete),
        "support_read_complete": int(read_complete),
        "support_passage_cited_complete": int(cited_complete),
        "original_sentence_read_complete": int(sentence_read),
        "original_sentence_cited_complete": int(sentence_cited),
        "answer_and_original_sentence_cited": int(
            sentence["answer_and_sentence_support_complete"]
        ),
        "read_sentence_fraction": (
            sentence["supporting_sentences_read"] / sentence["original_support_facts_count"]
        ),
        "cited_sentence_fraction": (
            sentence["supporting_sentences_cited"] / sentence["original_support_facts_count"]
        ),
        "source_lexical_candidate": int(src.lexical_candidate),
        "source_cross_document_phantom_candidate": int(src.concat_only_lexical_candidate),
        "support_read_recall": len(gold & read) / len(gold),
        "support_retrieval_recall": len(gold & exposed) / len(gold),
        "strict_sentence_and_em_but_not_entailment": int(
            sentence["answer_and_sentence_support_complete"]
        ),
        "semantic_entailment_evaluated": 0,
        "termination": str(trace.get("termination", "")),
        "search_calls": int(trace.get("search_calls", 0)),
        "read_calls": int(trace.get("read_calls", 0)),
        "tool_calls": int(trace.get("tool_calls", 0)),
        "model_calls": int(trace.get("model_calls", 0)),
        "prompt_tokens": int(trace.get("prompt_tokens", 0)),
        "completion_tokens": int(trace.get("completion_tokens", 0)),
        "elapsed_ms": int(trace.get("elapsed_ms", 0)),
    }
