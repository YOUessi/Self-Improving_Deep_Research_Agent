"""Conservative CPU checks for known false bridges in TRAIN semantic reviews.

A flag can reject a candidate, but no absence of flags can certify a new
answer. This is NOT a general natural-language entailment verifier.
"""
from __future__ import annotations

import re


def _has(text: str, phrase: str) -> bool:
    if not phrase or len(phrase) < 3:
        return False
    return bool(re.search(
        r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
        text, flags=re.IGNORECASE,
    ))


def negative_flags(question: str, answer: str, citations: tuple[dict, ...],
                   proposed_bridge: str) -> tuple[str, ...]:
    """Detect composite-name collisions, unsupported trophies, city swaps."""
    if len(citations) != 2 or len({x["doc_id"] for x in citations}) != 2:
        raise ValueError("two distinct original citations required")
    quotes = [x["quote"] for x in citations]
    flags: set[str] = set()
    bridge = proposed_bridge.strip()
    if bridge:
        for quote in quotes:
            for m in re.finditer(
                r"(?<!\w)" + re.escape(bridge) + r"(?!\w)",
                quote, flags=re.IGNORECASE,
            ):
                leading = quote[max(0, m.start() - 70):m.start()]
                if re.search(
                    r"[A-Z][A-Za-z'-]+\s*(?:&|and)\s*$", leading,
                ):
                    flags.add("bridge_component_of_larger_named_entity")
    joined = " ".join(quotes)
    if (
        re.search(r"\bchampionships?\b", question, re.IGNORECASE)
        and not re.search(
            r"\b(championships?|champions?|titles?|victories)\b",
            joined, re.IGNORECASE,
        )
    ):
        flags.add("championship_relation_absent_from_both_quotes")
    if (
        re.search(r"\b(population|census)\b", question, re.IGNORECASE)
        and re.search(r"\b(where|located|based|city)\b", question, re.IGNORECASE)
    ):
        evidence_for_answer = [x for x in citations if answer in x["quote"]]
        evidence_for_location = [x for x in citations if answer not in x["quote"]]
        if len(evidence_for_answer) == len(evidence_for_location) == 1:
            q = evidence_for_location[0]["quote"]
            locations = re.findall(
                r"\b([A-Z][a-zA-Z'-]+),\s+[A-Z][a-zA-Z'-]+\s+based\b",
                q,
            )
            if locations and not any(
                _has(evidence_for_answer[0]["quote"], location)
                for location in locations
            ):
                flags.add("company_base_city_not_in_population_evidence")
    return tuple(sorted(flags))
