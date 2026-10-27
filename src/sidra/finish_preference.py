"""TRAIN-only counterfactual FINISH preference on *real visited tool states*.

This is an OPTIONAL supervised auxiliary term for an online policy optimizer.
It is NOT part of the on-policy group-relative GRPO estimator: it uses
original TRAIN ground truth AFTER an episode has completed to identify the
correct FINISH among actions the model had actually been offered.

The important distinction: the policy itself never receives labels, gold
documents or a new oracle action. Supervision only chooses among nonoracle,
already-visible legal actions in a sampled state, for differentiable policy
training. A model still samples from exactly the same categorical logits
with the same v4 observation prior when online GRPO rollouts execute.

A future experiment MUST separately compare pure on-policy GRPO, supervised
FINISH-only, and their hybrid on new independent TRAIN and frozen public DEV.
"""
from __future__ import annotations

import math
from collections import Counter

from sidra.metrics import normalize_answer
from sidra.schema import Question


def annotate_train_finish_preferences(episodes:list[dict],question:Question)->dict:
    """Attach TRAIN verifier-only positive legal FINISH indices to sampled states.

    Only prompts that the model actually encountered are eligible. Require
    the complete ORIGINAL TRAIN supporting-doc IDs *already read* before
    that decision; no partial evidence, unvisited states or invented targets.
    The correct answer must appear verbatim after normalization in the text
    of one of the cited actual READ observations. Reject relabelled DEV/TEST
    and all sample-vs-trace mismatches rather than guessing silently.

    Labels go solely into the optimizer's action_samples, NOT into prompts,
    tool history, candidate proposer, prior or model-generated output.
    """
    if question.split!="train":
        raise ValueError("counterfactual FINISH needs original TRAIN labels")
    if not question.gold_answer.strip() or not question.supporting_doc_ids:
        raise ValueError("TRAIN answer and support documents required")
    if len(episodes)<2:
        raise ValueError("on-policy TRAIN episodes required for preference audit")
    gold=set(question.supporting_doc_ids)
    answer=normalize_answer(question.gold_answer)
    counts=Counter()
    for ep in episodes:
        tr=ep.get("trace")
        if ep.get("qid")!=question.question_id or not isinstance(tr,dict):
            raise ValueError("TRAIN episode identity mismatch")
        if tr.get("question_id")!=question.question_id or tr.get("question")!=question.question:
            raise ValueError("observed episode question mismatch")
        samples=ep.get("action_samples")
        steps=tr.get("steps")
        if not isinstance(samples,list) or not isinstance(steps,list) or len(samples)!=len(steps):
            raise ValueError("model-selected ACTIONs must match full REAL trace")
        read:dict[str,str]={}
        found:set[str]=set()
        for i,(sample,step) in enumerate(zip(samples,steps,strict=True)):
            choices=sample.get("legal_actions",[])
            selected=sample.get("selected_index")
            actual=step.get("action")
            if (sample.get("selection_mode")!="sample"
                or sample.get("action_token_targets")!=1
                or sample.get("tool_observation_targets")!=0
                or not isinstance(selected,int)
                or not 0<=selected<len(choices)
                or choices[selected]!=actual):
                raise ValueError(f"counterfactual uses only authentic ACTION samples: {i}")
            if "train_only_finish_positive_indices" in sample:
                raise ValueError("counterfactual annotation cannot overwrite previous labels")
            counts["visited_action_states"]+=1
            positives=[]
            if gold.issubset(read):
                counts["states_after_full_real_support_reads"]+=1
                for j,c in enumerate(choices):
                    if c.get("kind")!="finish" or not c.get("answer","").strip():
                        continue
                    cited=c.get("citations")
                    # Action.to_dict() keeps tuple citations in-memory;
                    # JSONL round-trips make them lists. Both are canonical.
                    if (not isinstance(cited,(list,tuple))
                        or len(cited)!=len(set(cited))
                        or not all(isinstance(doc_id,str) for doc_id in cited)):
                        raise ValueError("invalid candidate citation IDs")
                    cite_set=set(cited)
                    if not (gold.issubset(cite_set) and cite_set.issubset(read)):
                        continue
                    if normalize_answer(c["answer"])!=answer:
                        continue
                    evidence=" ".join(read[cid] for cid in cited)
                    if answer not in normalize_answer(evidence):
                        continue
                    positives.append(j)
            if positives:
                if not set(positives).issubset(set(range(len(choices)))):
                    raise AssertionError("positive target missing from actual model action catalog")
                sample["train_only_finish_positive_indices"]=positives
                counts["eligible_states"]+=1
                if selected in positives:
                    counts["student_chose_correct_finish"]+=1
                else:
                    counts["student_ignored_correct_finish"]+=1
            # Consume the actual observation only AFTER evaluating the
            # current decision. The student does not see future tool outputs.
            if actual.get("kind")=="search":
                obs=step.get("observation") or {}
                if obs.get("kind")!="search":
                    raise ValueError("SEARCH action has no actual tool output")
                for h in obs.get("hits",[]):
                    if isinstance(h,dict) and h.get("doc_id"):
                        found.add(str(h["doc_id"]))
            elif actual.get("kind")=="read":
                obs=step.get("observation") or {}
                did=actual.get("doc_id")
                if (obs.get("kind")!="read" or did!=obs.get("doc_id")
                    or did not in found or did in read):
                    raise ValueError("READ requires actual previously-searched support")
                read[did]=str(obs.get("text",""))
            elif actual.get("kind")=="finish":
                if i!=len(samples)-1:
                    raise ValueError("FINISH must terminate real SearchAgent episode")
            else:
                raise ValueError("unexpected ACTION in real trace")
        counts["episodes"]+=1
    return {
        "stage":"TRAIN_ONLY_POSTHOC_SUPERVISED_PREFERENCE_NOT_GRPO",
        "policy_model_gold_access":False,
        "labels_inserted_into_action_samples_only":True,
        "counterfactual_targets_are_existing_legal_actions":True,
        **dict(counts),
    }


def finish_positive_nll(model,sample:dict):
    """Masked categorical negative log probability mass of correct legal FINISHes.

    Uses precisely the same legal choice token set and additive observation
    prior as FiniteActionPolicy.decide(), not unconstrained model logits.
    Unselected-vocabulary and tool-observation tokens get NO loss. This is
    supervised TRAIN-only auxiliary NLL, never claimed as policy-gradient
    on sampled trajectories.
    """
    import torch
    positives=sample.get("train_only_finish_positive_indices")
    ids=sample["legal_choice_token_ids"]
    choices=sample["legal_actions"]
    if not isinstance(positives,list) or not positives:
        raise ValueError("no verified TRAIN finish target")
    if (len(ids)!=len(choices) or len(set(ids))!=len(ids)
        or any(not isinstance(j,int) or not 0<=j<len(ids) for j in positives)
        or len(set(positives))!=len(positives)):
        raise ValueError("malformed finite legal FINISH target")
    if any(choices[j].get("kind")!="finish" or not choices[j].get("answer") for j in positives):
        raise ValueError("TRAIN target is not a factual legal FINISH")
    if sample.get("action_token_targets")!=1 or sample.get("tool_observation_targets")!=0:
        raise ValueError("label masking violation")
    prompt=sample.get("prompt_ids")
    if not isinstance(prompt,list) or not prompt:
        raise ValueError("no authentic model prompt")
    tokens=torch.tensor([prompt],device=model.device,dtype=torch.long)
    logits=model(input_ids=tokens).logits[0,-1,:].float()
    subset=logits[torch.tensor(ids,device=logits.device)]
    biases=sample.get("action_logit_biases")
    if biases is not None:
        if (len(biases)!=len(ids) or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>4
            for v in biases)):
            raise ValueError("corrupt observation-only behavior prior")
        subset=subset+torch.tensor(biases,device=logits.device,dtype=subset.dtype)
    lp=torch.log_softmax(subset,dim=-1)
    return -torch.logsumexp(lp[positives],dim=0)
