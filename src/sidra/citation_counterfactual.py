"""M7 E-025: citation leave-one-out (LOCO) NLI falsification screening.

CPU-only, TRAIN-proposed experimental evidence check. Never called by the
SearchAgent policy; no ground-truth answers or supporting sentence indices
are passed into this module.

A confident multi-document MNLI prediction is NOT proof of a correct answer.
A stronger falsification experiment removes each *actually cited READ*
source while keeping all other evidence fixed. If MNLI remains highly
confident when a claimed multi-hop source disappears, that source's
necessity has NOT been demonstrated. This is an opt-in abstaining gate,
not an oracle validator or a trained evidence verifier.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from sidra.semantic_verifier import (
    NLIScorer,SemanticReview,actual_cited_reads,semantic_review,sentence_windows,
)


@dataclass(frozen=True)
class CitationAblation:
    status: str
    reason: str
    baseline_nli_status: str
    baseline_nli_entailment: float | None
    citation_ids: tuple[str, ...]
    ablated_entailments: dict[str,float]
    weakest_entailment_drop: float | None
    additional_nli_calls: int
    model_input_tokens_max: int | None
    gold_was_not_accessed: bool = True
    semantic_ground_truth: str = "not_established"

    def as_dict(self)->dict[str,Any]:
        return asdict(self)


def _token_length(scorer:NLIScorer,premise:str,hypothesis:str)->int|None:
    """Check actual tokenizer's PAIR input before the scorer can truncate it."""
    tok=getattr(scorer,"tokenizer",None)
    max_length=getattr(scorer,"max_length",None)
    if tok is None or max_length is None:
        return None
    ids=tok(premise,hypothesis,truncation=False)["input_ids"]
    # Some tokenizers return a nested batch, others a single vector.
    if ids and isinstance(ids[0],list):
        if len(ids)!=1:raise ValueError("NLI tokenizer gave more than one pair")
        ids=ids[0]
    return len(ids)


def _validate_posterior(row:dict[str,float])->float:
    if set(row)!={"entailment","neutral","contradiction"}:
        raise ValueError("NLI posterior missing a required label")
    if (not all(isinstance(v,(float,int)) and math.isfinite(v) and
                0<=v<=1 for v in row.values())
        or abs(sum(row.values())-1)>0.01):
        raise ValueError("NLI posterior invalid")
    return float(row["entailment"])


def citation_leave_one_out(
    trace:dict[str,Any], question:str,scorer:NLIScorer, *,
    base_review:SemanticReview|None=None,
    max_citations:int=3,
    entailment_threshold:float=0.85,
    removal_ceiling:float=0.65,
    min_entailment_drop:float=0.20,
)->CitationAblation:
    """Determine if *each* citation is necessary for current MNLI confidence.

    Preconditions:
      - Source-only evidence legality and baseline NLI have already passed;
      - All citations are authentic matched READ observations (not snippets);
      - Sentence windows in each source and in the full premise match exactly;
        this rejects context reallocation/capping artifacts;
      - When the actual tokenizer is available, all full/ablation NLI inputs
        must fit the model context WITHOUT truncation.

    Fixed gate (chosen on TRAIN, not DEV): p_full>=.85; for EACH removed cite,
    p_drop<=.65 AND p_full-p_drop>=.20. No adaptive threshold tuning on Pilot32.
    """
    if not 2<=max_citations<=6:raise ValueError("invalid max_citations")
    if not (0.5<=entailment_threshold<=1.0
            and 0<=removal_ceiling<entailment_threshold
            and 0<min_entailment_drop<1):
        raise ValueError("invalid counterfactual thresholds")
    review=base_review or semantic_review(
        trace,question,scorer,entailment_threshold=entailment_threshold)
    citations=tuple(map(str,trace.get("citations") or []))

    def finish(status:str,reason:str, *,
               scores:dict[str,float]|None=None,token_max:int|None=None,
               calls:int=0)->CitationAblation:
        values=scores or {}
        drop=(min(float(review.entailment)-p for p in values.values())
              if values and review.entailment is not None else None)
        return CitationAblation(
            status,reason,review.status,review.entailment,citations,
            values,drop,calls,token_max)

    if (review.status!="model_predicted_entailment"
        or review.entailment is None
        or review.entailment<entailment_threshold):
        return finish("not_eligible","baseline_not_confidently_predicted_entailment")
    if len(set(citations))!=len(citations) or not 2<=len(citations)<=max_citations:
        return finish("not_eligible","invalid_or_excessive_citation_count")
    reads=actual_cited_reads(trace)
    if any(cid not in reads for cid in citations):
        return finish("not_eligible","citation_missing_actual_read")
    hypothesis=review.hypothesis
    if not hypothesis:
        return finish("not_eligible","missing_baseline_hypothesis")

    answer=str(trace.get("answer",""))
    premise_all, n=sentence_windows(question,answer,reads,citations)
    if not premise_all or n<len(citations):
        return finish("not_eligible","missing_source_sentence_window")
    individual=[]
    for doc_id in citations:
        premise,_=sentence_windows(question,answer,reads,(doc_id,))
        if not premise:
            return finish("not_eligible","empty_cited_source_window")
        individual.append(premise)
    # Critical: each source's window must be identical in full and ablation
    # scoring. We must NEVER mistake context reallocation for source necessity.
    if "\n".join(individual)!=premise_all:
        return finish("not_eligible","sentence_window_budget_interaction")

    omissions={
        dropped:"\n".join(p for d,p in zip(citations,individual,strict=True)
                          if d!=dropped)
        for dropped in citations
    }
    inputs=[premise_all,*omissions.values()]
    token_lengths=[_token_length(scorer,p,hypothesis) for p in inputs]
    available=[n for n in token_lengths if n is not None]
    token_max=max(available) if available else None
    max_length=getattr(scorer,"max_length",None)
    if (available and max_length is not None
        and max(available)>int(max_length)):
        return finish("not_eligible","nli_input_would_truncate",token_max=token_max)

    # Do not re-score full premise; exact same windows and tokenizer checked.
    scored=scorer.score(list(omissions.values()),[hypothesis]*len(citations))
    if len(scored)!=len(citations):
        raise ValueError("NLI backend returned wrong ablation length")
    results={cid:_validate_posterior(row) for cid,row in zip(citations,scored,strict=True)}
    necessary=all(
        value<=removal_ceiling and review.entailment-value>=min_entailment_drop
        for value in results.values()
    )
    return finish(
        "counterfactual_necessary_model_prediction" if necessary
        else "blocked_evidence_nonessential",
        ("every_cited_source_changes_NLI_prediction" if necessary
         else "at_least_one_source_removal_preserves_high_entailment"),
        scores=results,token_max=token_max,calls=len(citations),
    )
