"""TRAIN-label-blind state-dependent prior for legal finite ACTION choices (v4).

The prior is a bounded, deterministic function of the question, actual
Search/Read observations and the already verified legal catalog. It does not
consult original documents not returned by tools, evaluator gold, or a
corpus-wide index. It is part of the *behavior policy*, not a reward model:
its additive logits MUST be included in the differentiable selected-action
log probability used for online GRPO-style updates.

Motivation: M7-A5 TRAIN traces showed 35/160 episodes had a gold support doc
as a legal READ candidate but never read it; 80/160 never searched all gold.
This prior is a hypothesis to be tested, not an established improvement.
"""
from __future__ import annotations

import math

from sidra.retrieval import tokenize
from sidra.schema import Action


def _overlap(a:set[str],b:set[str])->float:
    if not a or not b:
        return 0.0
    return len(a & b)/len(a)


def action_logit_biases(
    question:str,observations:list[dict],actions:list[Action],
)->list[float]:
    """Bounded v4 read/search decision priors using ONLY visible information.

    Every observation field is accessed by an explicit allowlist: 'kind',
    'query', 'hits', 'doc_id', 'title', 'snippet', 'text'. Any injected
    'gold_answer'/'supporting_doc_ids'/hidden labels are ignored.

    No tool budget is changed; the prior never changes the legal catalog,
    and never reads unseen corpus documents. The numeric prior belongs to
    the exact categorical sampling and gradient normalization.
    """
    if not question.strip() or not actions:
        raise ValueError("question and legal action catalog required")
    qterms=set(tokenize(question))
    read_obs=[o for o in observations if o.get("kind")=="read"]
    observed_reads=" ".join(str(o.get("text",""))[:1600] for o in read_obs).casefold()
    read_terms=set(tokenize(observed_reads))
    hits:dict[str,dict]={}
    for o in observations:
        if o.get("kind")!="search":
            continue
        for hit in o.get("hits",[]):
            if isinstance(hit,dict):
                did=str(hit.get("doc_id",""))
                if did:
                    hits.setdefault(did,hit)
    read_ids={str(o.get("doc_id","")) for o in read_obs}
    budgets_exhausted=not any(a.kind in {"search","read"} for a in actions)
    bias:list[float]=[]
    for a in actions:
        if a.kind=="read":
            if a.doc_id not in hits or a.doc_id in read_ids:
                raise ValueError("prior refuses an unseen/already-read document")
            hit=hits[a.doc_id]
            title=str(hit.get("title",""))[:150]
            title_terms=set(tokenize(title))
            snippet_terms=set(tokenize(str(hit.get("snippet",""))[:220]))
            # Word intersection is a *relevance heuristic* not gold support.
            question_match=_overlap(title_terms,qterms)
            snippet_match=_overlap(qterms,snippet_terms)
            bridge_match=_overlap(title_terms,read_terms) if read_obs else 0.0
            title_mentioned=(bool(title) and bool(read_obs)
                             and title.casefold() in observed_reads)
            stage_bonus=0.65 if len(read_obs)<2 else 0.1
            score=(0.25 + stage_bonus + 0.65*question_match
                   + 0.3*snippet_match + 0.65*bridge_match
                   + (0.85 if title_mentioned else 0.0))
        elif a.kind=="search":
            # Bridge searches proposed after a READ get a mild boost only
            # when their terms actually occur in read context or question.
            terms=set(tokenize(a.query))
            common=_overlap(terms,qterms|read_terms)
            score=0.1+0.30*common
        elif a.kind=="finish":
            if a.answer:
                # Avoid rewarding early finishing on superficial lexical
                # overlap; no attempt to guess correctness from hidden gold.
                score=0.0
            else:
                # Abstention should not crowd out real tool evidence while
                # search/read options remain. Never make it illegal.
                score=0.0 if budgets_exhausted else -0.35
        else:
            raise ValueError("unsupported action kind")
        if not math.isfinite(score) or abs(score)>4:
            raise ValueError("invalid bounded categorical prior")
        bias.append(round(float(score),6))
    return bias
