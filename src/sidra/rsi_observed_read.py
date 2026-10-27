"""E-029: gold-blind same-budget second-READ proposal from observed tool state.

This module deliberately imports NO trusted dataset, reward, evaluator, or
original TRAIN question mapping. Its output is an optional opaque document ID,
NOT a policy-gradient training sample, final answer, or full model rollout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sidra.retrieval import tokenize


@dataclass(frozen=True)
class ReadProposal:
    doc_id: str
    visible_candidates: int


def _validated_frontier(observations: list[dict]) -> tuple[dict[str, dict], set[str], str]:
    """Fail closed on any fields other than SearchAgent's actual observations."""
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    hits: dict[str, dict] = {}
    read: set[str] = set()
    observed_read_text: list[str] = []
    for event in observations:
        if not isinstance(event, dict):
            raise ValueError("observation must be a dict")
        if event.get("kind") == "search":
            if set(event) != {"kind", "query", "hits"}:
                raise ValueError("forbidden search observation fields")
            if not isinstance(event["query"], str) or not isinstance(event["hits"], list):
                raise ValueError("invalid search observation")
            for hit in event["hits"]:
                if not isinstance(hit, dict) or set(hit) != {
                    "doc_id", "title", "snippet", "score"
                }:
                    raise ValueError("forbidden search-hit fields")
                if (not all(isinstance(hit[x], str) for x in ("doc_id", "title", "snippet"))
                    or not hit["doc_id"] or type(hit["score"]) not in (int, float)
                    or not math.isfinite(hit["score"])):
                    raise ValueError("invalid observed search hit")
                old = hits.get(hit["doc_id"])
                if old and any(old[k] != hit[k] for k in ("title", "snippet")):
                    raise ValueError("conflicting search-hit provenance")
                # Use score of latest actual query, retain first-hit order.
                hits[hit["doc_id"]] = hit
        elif event.get("kind") == "read":
            if set(event) != {"kind", "doc_id", "title", "text", "truncated"}:
                raise ValueError("forbidden read observation fields")
            if (not all(isinstance(event[x], str) for x in ("doc_id", "title", "text"))
                or type(event["truncated"]) is not bool
                or event["doc_id"] not in hits
                or event["title"] != hits[event["doc_id"]]["title"]):
                raise ValueError("READ provenance missing from prior SEARCH")
            read.add(event["doc_id"])
            observed_read_text.append(event["text"])
        else:
            raise ValueError("unknown or untrusted tool observation")
    return hits, read, "\n".join(observed_read_text)


def choose_observed_second_read(
    question: str, observations: list[dict], *,
    max_steps: int = 6, max_tool_calls: int = 5
) -> ReadProposal | None:
    """Choose from *already-exposed* unread hits, with one READ+FINISH slot left.

    No gold, map or corpus arguments exist. Requires >=2 alternative legal
    unread hits to make a meaningful matched action-slot comparison.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if type(max_steps) is not int or type(max_tool_calls) is not int or (
        max_steps < 2 or max_tool_calls < 1
    ):
        raise ValueError("invalid original tool budgets")
    hits, read, evidence = _validated_frontier(observations)
    if not read:
        return None
    # Every Search/Read observation costs exactly one original tool call;
    # final Finish requires its own remaining model decision step.
    if len(observations) > max_steps - 2 or len(observations) >= max_tool_calls:
        return None
    unseen = [(doc_id, h) for doc_id, h in hits.items() if doc_id not in read]
    if len(unseen) < 2:
        return None
    question_words = set(tokenize(question))
    read_words = set(tokenize(evidence))
    folded_q = question.casefold()
    folded_e = evidence.casefold()

    def rank(item: tuple[str, dict]) -> tuple[float, int, str]:
        doc_id, hit = item
        title = hit["title"].strip()
        title_words = {t for t in tokenize(title) if len(t) >= 3}
        snippet_words = set(tokenize(hit["snippet"]))
        # First entity linked by a previous READ gets preference; question
        # entity grounding is next. Unseen full document text is unavailable.
        phrase_read = bool(title and len(title_words) >= 2 and title.casefold() in folded_e)
        phrase_question = bool(title and len(title_words) >= 2 and title.casefold() in folded_q)
        score = (
            5.0 * phrase_read
            + 3.5 * phrase_question
            + 1.6 * len(title_words & read_words)
            + 1.2 * len(title_words & question_words)
            + 0.15 * len(snippet_words & question_words)
            + 0.10 * len(snippet_words & read_words)
            + 0.04 * min(max(float(hit["score"]), 0.0), 20.0)
        )
        return (-score, list(hits).index(doc_id), doc_id)

    chosen = min(unseen, key=rank)[0]
    if chosen not in hits or chosen in read:
        raise AssertionError("selector output is not an observed legal READ")
    return ReadProposal(chosen, len(unseen))
