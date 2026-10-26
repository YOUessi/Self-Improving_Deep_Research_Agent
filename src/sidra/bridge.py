"""Gold-blind second-hop query proposal from *observed* first-hop text.

This is a retrieval-policy baseline, NOT an oracle/answer extractor. It sees
only the public question, search-hit titles and text actually returned by READ.
No hidden target documents, evaluator gold answers or full corpus index access.

A candidate question entity not yet represented in the read title is preferred.
Otherwise select named entities from the first read's question-relevant sentence.
The returned string must still be passed through the ordinary audited SEARCH
tool and consumes a normal search call.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sidra.retrieval import tokenize

NAME=re.compile(
    r"\b(?:[A-Z][\w.-]*|[A-Z]{2,})"
    r"(?:[\s,]+(?:[A-Z][\w.-]*|[A-Z]{2,}|and|of|the|St\.|de|in|on|at)){1,7}",
)
GENERIC={
 "united states","new york","great britain","american english","north america",
 "in addition","the united","the new","the first","the second",
 "the city","the town","south park","united kingdom",
}


@dataclass(frozen=True)
class BridgeCandidate:
    query:str
    source:str
    score:float


def normalize(text:str)->str:
    return " ".join(tokenize(text))


def phrases(text:str)->list[str]:
    matches=[]
    seen=set()
    for m in NAME.finditer(text):
        s=m.group().strip(" ,.;:\t\n")
        s=re.sub(r"\s+"," ",s)
        if len(s)<7 or len(s)>110:continue
        key=normalize(s)
        if key in seen or key in GENERIC or len(key.split())<2:continue
        seen.add(key)
        matches.append(s)
    return matches


def bridge_candidates(question:str,read_doc_title:str,read_doc_text:str,
                      prior_queries:set[str]|None=None)->list[BridgeCandidate]:
    prior={normalize(x) for x in (prior_queries or set())}
    already=normalize(read_doc_title)
    question_terms=set(tokenize(question))
    read_terms=set(tokenize(read_doc_text))
    pool=[]
    for source,text,limit in [
        ("question_other_entity",question,20),
        ("read_entity",read_doc_text[:2800],80),
    ]:
        for s in phrases(text)[:limit]:
            key=normalize(s)
            if not key or key in prior or key==already:continue
            if key in already and len(key.split())>1:continue
            tokens=set(tokenize(s))
            if not tokens:continue
            if source=="question_other_entity" and len(tokens&read_terms)==0:
                # Still allowed comparative second entities, but only if the
                # question contains it, never from evaluator annotations.
                affinity=0.5
            else:
                affinity=len(tokens&question_terms)
            rarity=len(tokens)
            if source=="read_entity":
                # An entity in a sentence related to the question is more
                # plausible as bridge than an arbitrary capitalized aside.
                sentence_start=max(0,read_doc_text.rfind(".",0,
                    read_doc_text.find(s)+1)+1)
                sentence=read_doc_text[sentence_start:
                                       read_doc_text.find(s)+len(s)+150]
                affinity+=min(3,len(set(tokenize(sentence))&question_terms))
            score=2.0*(source=="question_other_entity")+float(affinity)+min(rarity,6)*0.3
            pool.append(BridgeCandidate(s,source,score))
    return sorted(pool,key=lambda x:(-x.score,-len(x.query),x.query.casefold()))


def next_bridge_query(question:str,observations:list[dict],
                      prior_queries:set[str]|None=None)->BridgeCandidate|None:
    reads=[o for o in observations if o.get("kind")=="read" and o.get("text")]
    if not reads:return None
    last=reads[-1]
    if len(reads)>2:return None
    prior=prior_queries or {
        str(o.get("query","")) for o in observations if o.get("kind")=="search"
    }
    candidates=bridge_candidates(question,str(last.get("title","")),
                                  str(last["text"]),prior_queries=prior)
    return candidates[0] if candidates else None
