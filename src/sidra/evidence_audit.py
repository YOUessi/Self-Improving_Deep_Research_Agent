"""Non-oracle evidence audit for role-separated research agents."""

from __future__ import annotations
from dataclasses import dataclass
from sidra.evidence_diagnostics import _present_in_one_document
from sidra.schema import Action


@dataclass(frozen=True)
class Audit:
    approved: bool
    reason: str


class EvidenceAudit:
    def __init__(self, min_citations: int = 2):
        if min_citations < 1:
            raise ValueError("min_citations must be positive")
        self.min_citations = min_citations

    def verify(self, candidate: Action, observations: list[dict]) -> Audit:
        if candidate.kind != "finish" or not candidate.answer.strip():
            return Audit(False, "missing_answer")
        read = {o["doc_id"]: o.get("text", "") for o in observations if o.get("kind") == "read"}
        ids = candidate.citations
        if len(set(ids)) < self.min_citations or len(set(ids)) != len(ids):
            return Audit(False, "missing_or_duplicate_citations")
        if not set(ids).issubset(read):
            return Audit(False, "unread_citations")
        # Do not join READ passages before matching. Previous concatenation
        # could turn "New" in document A + "York" in document B into a
        # fabricated lexical answer; raw substring can also accept "York"
        # inside "Yorkshire". This is lexical support only, not entailment.
        if not any(_present_in_one_document(candidate.answer, read[i])
                   for i in ids):
            return Audit(False, "answer_not_in_sources")
        return Audit(True, "lexical_only")
