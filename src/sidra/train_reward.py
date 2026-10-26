"""TRAIN-only rollout rewards for future multi-turn Agentic GRPO.

This module DOES NOT perform RL training. Scores completed search/read/finish
trajectories with TRAIN-split gold labels and nonoracle tool observations.
It must never receive evaluation dev/test examples or hidden labels.

An outcome-complete answer+evidence score dominates syntax / efficiency to
avoid reward hacking by cheap abstentions or repeated search. The environment
runner (SearchAgent) determines tool legality. Semantic entailment beyond
passage-level support is not yet established.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from sidra.metrics import score_trace
from sidra.schema import AgentTrace, Document, Question


@dataclass(frozen=True)
class SearchRolloutReward:
    total: float
    execution_valid: float
    answer_correct: float
    cited_support_complete: float
    read_support_recall: float
    strict_outcome: float
    efficiency_bonus: float

    def as_dict(self)->dict:
        return {name:getattr(self,name) for name in self.__dataclass_fields__}


def score_train_rollout(q:Question, trace:AgentTrace,
                        documents:dict[str,Document], *,
                        max_tool_calls:int=5)->SearchRolloutReward:
    if q.split!="train":
        raise ValueError("TRAIN-only rewards reject dev/test labels")
    if q.question_id!=trace.question_id:
        raise ValueError("rollout question ID mismatch")
    if max_tool_calls<1:
        raise ValueError("tool budget must be positive")
    m=score_trace(q,trace,documents)
    gold=set(q.supporting_doc_ids)
    actually_read=set()
    for step in trace.steps:
        o=step.get("observation") or {}
        if o.get("kind")=="read":
            actually_read.add(str(o.get("doc_id","")))
    recalled=len(actually_read & gold)/len(gold) if gold else 0.
    cited=set(trace.citations)
    citation_ok=bool(gold) and gold.issubset(cited) and cited.issubset(actually_read)
    execution_valid=float(trace.termination not in {"error"})
    answer_correct=float(m["exact_match"])
    fully_verified=float(m["verified_success"] and citation_ok and
                        bool(m["answer_in_cited_evidence"]))
    # Efficiency points require *successful* evidence-backed answer, thus a
    # shortcut (finish empty or without citations) earns no cheap bonus.
    cost=float(fully_verified*max(0.,1.-trace.tool_calls/max_tool_calls))
    total=(0.05*execution_valid
           +0.55*fully_verified
           +0.15*answer_correct
           +0.10*float(citation_ok)
           +0.10*recalled
           +0.05*cost)
    return SearchRolloutReward(
        total=round(total,6),
        execution_valid=execution_valid,
        answer_correct=answer_correct,
        cited_support_complete=float(citation_ok),
        read_support_recall=round(recalled,6),
        strict_outcome=fully_verified,
        efficiency_bonus=round(cost,6),
    )


def group_normalized_advantages(rewards:list[float], *,
                                eps:float=1e-6)->list[float]:
    """GRPO-style group centering; no optimizer or rollout generation here."""
    if not rewards:
        raise ValueError("empty rollout group")
    if eps<=0 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("invalid reward group")
    avg=sum(rewards)/len(rewards)
    var=sum((x-avg)**2 for x in rewards)/len(rewards)
    if var<=eps*eps:
        return [0.0]*len(rewards)
    stdev=math.sqrt(var)
    return [(x-avg)/stdev for x in rewards]
