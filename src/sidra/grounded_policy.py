"""Evidence-aware action guard for a *single* existing LLM policy.

No gold answer, supporting passages or evaluator output are available here.
The guard prevents repeated reads/queries, repairs cite-before-read mistakes
by reading previously *retrieved* documents, and gathers more than one source
before finishing when a bounded tool budget permits. It never fabricates
citations or fills in an answer from evaluation labels.

All model calls (including optional one-shot citation revision) and tool calls
are still measured by the existing SearchAgent. Study its reward/latency cost.
"""
from __future__ import annotations

from sidra.policies import SYSTEM
from sidra.schema import Action


class GroundedPolicy:
    def __init__(self, base, *, min_reads: int = 2, revise_citations: bool = True):
        if min_reads < 1:
            raise ValueError("min_reads must be >= 1")
        self.base = base
        self.name = "grounded_" + base.name
        self.min_reads = min_reads
        self.revise_citations = revise_citations
        self.model_name = getattr(base, "model_name", getattr(base, "model", None))
        self.adapter_path = getattr(base, "adapter_path", None)
        self.last_role = ""
        self.last_verification = None
        self.last_model_calls = 0

    @property
    def model_calls(self) -> int:
        return int(getattr(self.base, "model_calls", 0))

    @property
    def prompt_tokens(self) -> int:
        return int(getattr(self.base, "prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(getattr(self.base, "completion_tokens", 0))

    def _decide(self, question: str, observations: list[dict], remaining: int,
                extra: str = "") -> Action:
        role = getattr(self.base, "decide_with_system", None)
        self.last_role = "grounded_policy"
        instruction = (
            "Source discipline: A search hit is NOT a read source. "
            "Before answering, read two distinct relevant documents when available. "
            "Only cite IDs observed in previous READ results; never cite snippets alone. "
            "Give a short factual answer, not a description of the question. "
            "Avoid duplicate queries and repeat reading. "
            "Return ONE JSON action, never unsupported citations."
        )
        if callable(role):
            result = role(question, observations, remaining,
                          SYSTEM + "\n" + instruction + ("\n" + extra if extra else ""))
        else:
            result = self.base.decide(question, observations, remaining)
        if not isinstance(result, Action):
            raise TypeError("GroundedPolicy backend must return Action")
        return result

    @staticmethod
    def _seen(observations: list[dict]) -> tuple[set[str], set[str], list[dict]]:
        read = {str(o["doc_id"]) for o in observations if
                o.get("kind") == "read" and o.get("doc_id")}
        queries = {str(o["query"]).strip().casefold() for o in observations
                   if o.get("kind") == "search" and o.get("query")}
        hits: dict[str, dict] = {}
        # Keep first occurrence with its original rank/order to preserve a
        # stable frontier even when the model searches the same query again.
        for o in observations:
            if o.get("kind") == "search":
                for hit in o.get("hits", []):
                    hits.setdefault(str(hit["doc_id"]), hit)
        return read, queries, list(hits.values())

    @staticmethod
    def _unread(frontier: list[dict], read: set[str]) -> list[str]:
        return [str(hit["doc_id"]) for hit in frontier if str(hit["doc_id"]) not in read]

    def _override(self, action: Action, reason: str) -> Action:
        self.last_verification = {"approved": False, "reason": reason}
        return action

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        self.last_verification = None
        self.last_model_calls = 0
        before = self.model_calls
        read, queries, frontier = self._seen(observations)
        action = self._decide(question, observations, remaining)
        # Total calls, not just number of tool-level policy steps.
        self.last_model_calls = self.model_calls - before
        unread = self._unread(frontier, read)

        if action.kind == "read":
            if action.doc_id in read and remaining > 1 and unread:
                return self._override(Action("read", doc_id=unread[0]),
                                      "duplicate_read_repaired")
            if action.doc_id not in {str(h["doc_id"]) for h in frontier} and remaining > 1 and unread:
                return self._override(Action("read", doc_id=unread[0]),
                                      "read_missing_search_hit_repaired")
            return action

        if action.kind == "search":
            if action.query.strip().casefold() in queries and remaining > 1 and unread:
                return self._override(Action("read", doc_id=unread[0]),
                                      "repeated_query_frontier_read")
            return action

        if action.kind != "finish" or not action.answer.strip():
            return action

        # Cite-before-read is common: instead of letting the runner error,
        # read only a source already exposed in search observations.
        unseen_cites = [d for d in action.citations if d not in read]
        hit_ids = {str(h["doc_id"]) for h in frontier}
        for doc_id in unseen_cites:
            if doc_id in hit_ids and remaining > 1:
                return self._override(Action("read", doc_id=doc_id),
                                      "unread_citation_repaired")

        # At least two independent actual reads when available, with the
        # same max_steps / max_tool_calls constraints as the baseline.
        if len(read) < self.min_reads and unread and remaining > 1:
            return self._override(Action("read", doc_id=unread[0]),
                                  "finish_deferred_for_evidence")

        # If model emitted a one-source answer after reading >=2 documents,
        # allow ONE extra inference call, requesting a grounded re-evaluation.
        if (self.revise_citations and len(read) >= self.min_reads and
                len(set(action.citations)) < self.min_reads and remaining > 1):
            revised = self._decide(question, observations, remaining,
                "You have READ the following distinct document IDs: "
                + ", ".join(sorted(read))
                + ". Cite all relevant READ documents, at least two if they"
                  " jointly support the answer; if evidence remains insufficient"
                  " choose search/read instead of fabricating a source.")
            self.last_model_calls = self.model_calls - before
            if revised.kind == "finish" and (set(revised.citations) <= read) and (
                    len(set(revised.citations)) >= self.min_reads):
                return self._override(revised, "citation_revision_accepted")
            if revised.kind in {"read", "search"}:
                return self._override(revised, "citation_revision_more_research")
            # Do not erase correct one-source answers merely to inflate
            # citation counts; independent evaluator still scores them.
            self.last_verification = {
                "approved": False, "reason": "citation_revision_unsuccessful"}
        return action
