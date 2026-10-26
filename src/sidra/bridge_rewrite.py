"""Experimental, TRAIN-first second-hop relation-tail query rewrite.

No hidden index/corpus neighbors, label, support ID or gold answer is read by
this strategy. The only extra input beyond M7 FrontierFirst is the original
question's interrogative relation tail and one actually READ source title.
Rewrites only an ALREADY LEGAL/billed SEARCH plan; it cannot change budgets,
forge a READ ID, or synthesize a FINISH answer.
"""
from __future__ import annotations

import re

from sidra.bridge import normalize
from sidra.next_hop_policy import NextHopPlan
from sidra.retrieval import tokenize
from sidra.schema import Action


_WH_WORDS={"who","what","which","where","when","whose"}
_STOP={
    "and","are","also","being","been","have","has","had","was","were",
    "with","from","onto","into","that","this","there","these","those",
    "your","their","them","then","than","whom","much","many","more",
    "some","first","last","name","names","called","known","named",
    "title","does","did","would","could","should","following",
    "among","about","what","which","where","when","whose","that",
}


def relation_tail(question:str,bridge_query:str,first_read_title:str, *,
                  max_tokens:int=3)->str:
    """Question-text-only relation cue to disambiguate an observed bridge.

    Fixed BEFORE consulting any DEV labels. The interrogative tail normally
    contains e.g. "Austrian statesman" rather than the already-read document
    title. Do not guess missing synonyms or query external corpora.
    """
    if not 1<=max_tokens<=4:
        raise ValueError("max_tokens must be 1..4")
    # BM25 tokenizer removes stop words such as "what"; for this task we
    # MUST preserve interrogatives before deciding where the relation starts.
    toks=re.findall(r"[a-z][a-z0-9'-]*",question.casefold())
    indices=[i for i,t in enumerate(toks) if t in _WH_WORDS]
    tail=toks[indices[-1]+1:] if indices else toks[-7:]
    excluded=set(tokenize(bridge_query))|set(tokenize(first_read_title))
    suffix=[]
    for t in tail:
        if (len(t)>=4 and t not in _STOP and t not in excluded
                and not t.isdigit() and t not in suffix):
            suffix.append(t)
    return " ".join(suffix[-max_tokens:])


def rewrite_billed_second_search(
    plan:NextHopPlan|None,question:str,observations:list[dict],
) -> NextHopPlan|None:
    if plan is None or plan.action.kind!="search":
        return plan
    reads=[o for o in observations if o.get("kind")=="read"
           and o.get("doc_id")]
    if len(reads)!=1:
        return plan
    title=str(reads[0].get("title",""))
    suffix=relation_tail(question,plan.action.query,title)
    if not suffix:
        return plan
    query=plan.action.query+" "+suffix
    # Reject a duplicate query; it still costs SEARCH in SearchAgent.
    previous={normalize(str(o.get("query",""))) for o in observations
              if o.get("kind")=="search"}
    if normalize(query) in previous:
        return plan
    return NextHopPlan(
        Action("search",query=query),
        "observed_question_relation_tail_billed_search",
        plan.candidate,
    )
