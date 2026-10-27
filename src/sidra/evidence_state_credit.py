"""TRAIN-only per-step evidence-state credit for REAL SearchAgent rollouts.

This augments existing original TRAIN-gold search/read/answer delta attribution
with the *same* observation-only evidence-state potential used at inference.

Trust separation:
- FiniteActionPolicy sees only question and tool observations, never q.gold.
- AFTER the full episode, TRAIN-only scorer measures if actual newly
  searched/read sources were labelled gold and whether FINISH was correct.
- Add a small 0.25 observed-state potential delta to SOURCE actions; never
  credit mere abstention as progress. For outcome FINISH, use verified gold.
- Group normalization stays within the original QA and stage ordinal
  across real on-policy sibling rollouts (no pseudo-Oracle training trace).

Potential is a heuristic; a distractor READ may increase it. This is
a custom GRPO-style credit ablation, not a proof of policy invariance,
semantic entailment, or convergence of standard TRL GRPO.
"""
from __future__ import annotations

from collections import defaultdict
import math

from sidra.evidence_state import derive_evidence_state
from sidra.schema import Question
from sidra.step_credit import assign_step_evidence_credit,STAGES
from sidra.train_reward import group_normalized_advantages


def assign_evidence_state_credit(
    episodes:list[dict],question:Question, *,
    state_delta_weight:float=0.25,
    state_version:str="v1",
)->dict:
    if state_version not in {"v1","v2","v3"}:
        raise ValueError("unsupported observable evidence state version")
    if question.split!="train":
        raise ValueError("STATE reward with TRAIN gold refuses DEV/TEST")
    if not math.isfinite(state_delta_weight) or not 0<=state_delta_weight<=0.5:
        raise ValueError("invalid observed-state delta weight")
    base=assign_step_evidence_credit(episodes,question)
    # The existing validator checks exact TRAIN question and that sampled
    # legal action == actually executed SearchAgent action in trace.steps.
    groups=defaultdict(list)
    potential_stats={
        kind:{"actions":0,"nonzero_state_delta":0,
              "observed_progress_raw_total":0.0,"original_train_gold_raw_total":0.0,
              "changed_advantages":0,"credited_state_delta_total":0.0,
              "suppressed_state_delta_total":0.0}
        for kind in STAGES
    }
    for ep in episodes:
        obs_history=[]
        for sample,step in zip(ep["action_samples"],ep["trace"]["steps"],strict=True):
            kind=step["action"]["kind"]
            before=derive_evidence_state(question.question,obs_history,
                                          version=state_version)
            obs=step.get("observation")
            if obs is not None:
                obs_history.append(obs)
            after=derive_evidence_state(question.question,obs_history,
                                         version=state_version)
            delta=max(0.,after.potential-before.potential) if kind!="finish" else 0.0
            train_gold=float(sample["step_credit_raw"])
            # V3 reward is JOINT: observable graph delta is credited only
            # when the actual TRAIN support frontier also advanced. A title
            # coincidence from a distractor can NEVER be a reward by itself.
            # This original gold label is inspected AFTER real tool execution.
            credited_delta=(delta if state_version!="v3" or train_gold>0 else 0.0)
            if state_version=="v3" and kind=="finish":
                verified=float(ep["train_reward"]["strict_outcome"])
                correct=float(ep["train_reward"]["answer_correct"])
                score=0.2*correct+0.8*verified
            else:
                score=((1-state_delta_weight)*train_gold+
                       state_delta_weight*credited_delta
                       if kind!="finish" else train_gold)
            if not math.isfinite(score):
                raise ValueError("nonfinite TRAIN evidence state credit")
            sample["evidence_state_before"]=before.public_summary()
            sample["evidence_state_after"]=after.public_summary()
            sample["evidence_state_delta_raw"]=round(delta,6)
            sample["evidence_state_delta_credited_raw"]=round(credited_delta,6)
            sample["step_state_train_reward_raw"]=round(score,6)
            stage=(kind,int(sample["step_credit_ordinal"]))
            groups[stage].append((sample,score))
            stats=potential_stats[kind]
            stats["actions"]+=1
            stats["nonzero_state_delta"]+=int(delta>1e-8)
            stats["observed_progress_raw_total"]+=delta
            stats["credited_state_delta_total"]+=credited_delta
            stats["suppressed_state_delta_total"]+=delta-credited_delta
            stats["original_train_gold_raw_total"]+=train_gold
    total_nonzero=0
    for (kind,ordinal),rows in sorted(groups.items()):
        values=[x[1] for x in rows]
        adv=(group_normalized_advantages(values) if len(rows)>=2
             else [0.0]*len(rows))
        for (sample,_),value in zip(rows,adv,strict=True):
            if abs(value-sample["step_credit_advantage"])>1e-8:
                potential_stats[kind]["changed_advantages"]+=1
            sample["step_credit_advantage"]=float(value)
            total_nonzero+=int(abs(value)>1e-8)
    return {
        "mode":"evidence_state_delta_"+state_version,
        "train_question_id":question.question_id,
        "sibling_rollouts":len(episodes),
        "state_delta_weight":state_delta_weight,
        "state_version":state_version,
        "nonzero_optimizer_action_advantages":total_nonzero,
        "gold_source":"original TRAIN only, after real tool execution",
        "observable_state_source":"question + actual SEARCH/READ observations only",
        "policy_prompt_or_catalog_sees_original_labels":False,
        "tool_observation_label_tokens":0,
        "by_kind":potential_stats,
        "original_gold_step_credit":base,
        "science_limit":("V3 state delta is TRAIN-label gated AFTER execution; "
                         "at inference the link still proves only title co-mention, "
                         "not semantic entailment or generalization. "
                         "V1/V2 ungated behavior is unchanged."),
    }
