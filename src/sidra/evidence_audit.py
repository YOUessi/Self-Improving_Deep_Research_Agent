"""Non-oracle evidence audit for role-separated research agents."""

from __future__ import annotations
from dataclasses import dataclass
from sidra.metrics import normalize_answer
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
        text = normalize_answer(" ".join(read[i] for i in ids))
        if normalize_answer(candidate.answer) not in text:
            return Audit(False, "answer_not_in_sources")
        return Audit(True, "lexical_only")
