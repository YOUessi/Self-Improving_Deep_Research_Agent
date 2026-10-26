"""Per-decision TRAIN-only evidence-delta credit for REAL on-policy tool rollouts.

This is an exploratory custom group-relative masked policy-gradient reward, NOT
a TRL GRPOTrainer implementation and NOT a policy/oracle answer proposer.

Important trust boundary: model prompts and nonoracle legal ACTION catalogs are
already frozen by FiniteActionPolicy. Only after a full SearchAgent rollout does
this scorer inspect TRAIN Question.supporting_doc_ids / TRAIN answer metrics.
It annotates *optimizer-only* action_samples, never tool observations or the
model-visible prompt. DEV/TEST/relabelled challenger questions are rejected.

Search: first discovery of a TRAIN gold document among REAL returned BM25 hits.
Read: first actual reading of a TRAIN gold document.
Finish: correctness + strict cited evidence from the final TRAIN-only reward.

Group normalization is per ACTION kind, within the SAME original TRAIN question,
across independently sampled complete tool trajectories. In contrast to the old
stage_credit_v1, an irrelevant late read doesn't inherit success credit earned
by an earlier useful read in its episode.
"""
from __future__ import annotations

import math
from collections import defaultdict

from sidra.schema import Question
from sidra.train_reward import group_normalized_advantages

STAGES=("search","read","finish")


def assign_step_evidence_credit(episodes:list[dict], question:Question)->dict:
    """Attach a normalized credit to EACH actually sampled, executed ACTION.

    Raises on any mismatched action/trace sample, including output that might
    otherwise be silently scored against a different training question. Entire
    trajectories are on-policy samples: no ground truth influences sampling.
    """
    if question.split!="train":
        raise ValueError("step evidence credit strictly requires TRAIN labels")
    if len(episodes)<2:
        raise ValueError("step evidence credit requires >=2 real episodes")
    gold=set(question.supporting_doc_ids)
    if not gold:
        raise ValueError("step evidence credit requires original TRAIN supporting IDs")
    all_actions:dict[str,list[tuple[dict,float]]]=defaultdict(list)
    for ep in episodes:
        if ep.get("qid")!=question.question_id:
            raise ValueError("episode / TRAIN question ID mismatch")
        tr=ep.get("trace")
        if (not isinstance(tr,dict) or tr.get("question_id")!=question.question_id
            or tr.get("question")!=question.question):
            raise ValueError("missing or inconsistent real SearchAgent trace")
        samples=ep.get("action_samples",[])
        steps=tr.get("steps",[])
        if not samples or len(samples)!=len(steps):
            raise ValueError("every sampled ACTION must match one recorded tool step")
        reward=ep.get("train_reward",{})
        if (set(("answer_correct","strict_outcome"))-reward.keys()
            or not all(reward[k] in (0.,1.,0,1)
                       for k in ("answer_correct","strict_outcome"))):
            raise ValueError("TRAIN-only verified answer metrics missing")
        if reward["strict_outcome"] and not reward["answer_correct"]:
            raise ValueError("strict success must imply original answer correctness")
        found:set[str]=set()
        read:set[str]=set()
        finish_count=0
        for i,(sample,step) in enumerate(zip(samples,steps,strict=True)):
            action=step.get("action")
            choices=sample.get("legal_actions",[])
            sel=sample.get("selected_index")
            if (not isinstance(action,dict) or not isinstance(sel,int)
                or not 0<=sel<len(choices) or choices[sel]!=action
                or sample.get("action_token_targets")!=1
                or sample.get("tool_observation_targets")!=0
                or sample.get("selection_mode")!="sample"):
                raise ValueError(f"sample/trace/action-masking mismatch at {i}")
            kind=action.get("kind")
            if kind not in STAGES:
                raise ValueError("unexpected tool stage")
            obs=step.get("observation") or {}
            score=0.
            if kind=="search":
                if obs.get("kind")!="search":
                    raise ValueError("SEARCH was not really executed")
                hits=obs.get("hits",[])
                if not isinstance(hits,list):
                    raise ValueError("malformed actual search hits")
                current={str(h["doc_id"]) for h in hits
                         if isinstance(h,dict) and h.get("doc_id")}
                score=len((current-found)&gold)/len(gold)
                found.update(current)
            elif kind=="read":
                if obs.get("kind")!="read" or obs.get("doc_id")!=action.get("doc_id"):
                    raise ValueError("READ was not really executed")
                did=str(action["doc_id"])
                score=float(did in gold and did not in read)/len(gold)
                read.add(did)
            elif kind=="finish":
                if i!=len(steps)-1:
                    raise ValueError("FINISH must terminate the rollout")
                finish_count+=1
                # Correctness requires ORIGINAL TRAIN label; passage-level
                # support alone must not make a wrong answer a positive.
                score=(0.4*float(reward["answer_correct"])
                       +0.6*float(reward["strict_outcome"]))
            if not math.isfinite(score) or not 0<=score<=1.000001:
                raise ValueError("invalid step score")
            all_actions[kind].append((sample,score))
        if finish_count>1:
            raise ValueError("one episode contains multiple terminal actions")
    report={"credit_mode":"step_evidence_delta_v2","groups":len(episodes),
            "gold_support_source":"original TRAIN only",
            "policy_or_action_catalog_uses_gold":False,
            "by_kind":{}}
    for kind in STAGES:
        pairs=all_actions[kind]
        if not pairs:
            report["by_kind"][kind]={"actions":0,"positive_raw":0,
                                    "nonzero_advantages":0,"raw_mean":0.}
            continue
        scores=[score for _,score in pairs]
        advances=group_normalized_advantages(scores)
        for (sample,score),adv in zip(pairs,advances,strict=True):
            sample["step_credit_raw"]=float(score)
            sample["step_credit_advantage"]=float(adv)
        report["by_kind"][kind]={
            "actions":len(pairs),
            "positive_raw":sum(s>0 for s in scores),
            "nonzero_advantages":sum(abs(v)>1e-8 for v in advances),
            "raw_mean":round(sum(scores)/len(scores),6),
        }
    report["credited_actions"]=sum(
        v["nonzero_advantages"] for v in report["by_kind"].values())
    report["assistant_action_tokens"]=sum(
        v["actions"] for v in report["by_kind"].values())
    return report
