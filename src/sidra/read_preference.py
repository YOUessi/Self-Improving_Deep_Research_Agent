"""TRAIN-only counterfactual supporting-READ preference on REAL sampled states.

Separate from pure online GRPO: ORIGINAL TRAIN supporting-doc IDs act as
post-rollout *optimizer-side supervision*, NEVER as runtime legal-catalog
features, model prompt text, tool search hints or an artificial trajectory.

Purpose: A7 FINISH-only auxiliary correlated with fewer actual READ calls and
negative public-DEV original-support recall. This optional auxiliary tests a
specific alternative: favor an already-SEARCH-exposed, legally READ-able
unread supporting document *while evidence is incomplete*. It does not create
additional searches, bypass tool limits, invent gold READs, or supply unseen
documents.

An actual model policy selects its action BEFORE the label is attached.
"""
from __future__ import annotations

import math
from collections import Counter

from sidra.schema import Question


def annotate_train_read_preferences(episodes:list[dict],question:Question)->dict:
    """Label only legal READ candidates for missing original TRAIN support.

    The positives are indices into the *real catalog at that decision point*,
    not new oracle actions. SEARCH tool results are consumed after the selected
    action (never from a future observation). A gold ID must already have
    occurred in a real earlier search hit and be unread; otherwise the
    candidate is NOT supervised as an available READ.

    This is explicit supervised preference data. Never call it GRPO reward.
    """
    if question.split!="train" or not question.supporting_doc_ids:
        raise ValueError("read preference requires original TRAIN support labels")
    if len(episodes)<2:
        raise ValueError("on-policy group needs multiple real sampled episodes")
    gold=set(question.supporting_doc_ids)
    counts=Counter()
    for ep in episodes:
        trace=ep.get("trace")
        if ep.get("qid")!=question.question_id or not isinstance(trace,dict):
            raise ValueError("original TRAIN identity mismatch")
        if (trace.get("question_id")!=question.question_id
            or trace.get("question")!=question.question):
            raise ValueError("original TRAIN text mismatch")
        samples=ep.get("action_samples")
        steps=trace.get("steps")
        if (not isinstance(samples,list) or not isinstance(steps,list)
            or len(samples)!=len(steps)):
            raise ValueError("real selected ACTIONs differ from actual tool trace")
        found=set()
        read=set()
        for i,(sample,step) in enumerate(zip(samples,steps,strict=True)):
            actions=sample.get("legal_actions")
            selected=sample.get("selected_index")
            action=step.get("action")
            if (sample.get("selection_mode")!="sample"
                or sample.get("tool_observation_targets")!=0
                or sample.get("action_token_targets")!=1
                or not isinstance(actions,list)
                or not isinstance(selected,int) or isinstance(selected,bool)
                or not 0<=selected<len(actions)
                or actions[selected]!=action):
                raise ValueError(f"read preference needs authentic model ACTION {i}")
            if "train_only_read_positive_indices" in sample:
                raise ValueError("read preference cannot overwrite original annotations")
            positives=[]
            # Only the original support IDs that are ALREADY exposed as
            # legal READs in the student's true recorded state.
            if gold-read:
                for j,candidate in enumerate(actions):
                    if (candidate.get("kind")=="read"
                        and candidate.get("doc_id") in gold
                        and candidate.get("doc_id") in found
                        and candidate.get("doc_id") not in read):
                        positives.append(j)
            if positives:
                sample["train_only_read_positive_indices"]=positives
                counts["eligible_real_read_states"]+=1
                counts["positive_legal_read_options"]+=len(positives)
                if selected in positives:
                    counts["student_selected_support_read"]+=1
                else:
                    counts["student_ignored_offered_support_read"]+=1
            # Apply only the observation from the action ACTUALLY selected.
            if action["kind"]=="search":
                result=step.get("observation") or {}
                if result.get("kind")!="search":
                    raise ValueError("missing actual SEARCH observation")
                hits=result.get("hits")
                if not isinstance(hits,list):
                    raise ValueError("invalid actual SEARCH result")
                for hit in hits:
                    if isinstance(hit,dict) and isinstance(hit.get("doc_id"),str):
                        found.add(hit["doc_id"])
            elif action["kind"]=="read":
                result=step.get("observation") or {}
                did=action.get("doc_id")
                if (result.get("kind")!="read" or did!=result.get("doc_id")
                    or not isinstance(did,str) or did not in found or did in read):
                    raise ValueError("READ must follow real SEARCH and not be duplicated")
                read.add(did)
            elif action["kind"]=="finish":
                if i!=len(samples)-1:
                    raise ValueError("FINISH must terminate real trajectory")
            else:
                raise ValueError("unsupported real action")
            counts["visited_action_states"]+=1
        counts["episodes"]+=1
    return {
        "stage":"TRAIN_ONLY_POSTHOC_READ_SUPERVISION_NOT_GRPO",
        "model_visible_train_gold":False,
        "created_any_actions":False,
        "derived_from_original_train_support_only":True,
        "positive_indices_only_in_actual_legally_exposed_catalog":True,
        **dict(counts),
    }


def read_positive_nll(model,sample:dict):
    """NLL of already-legal original TRAIN positive READ choice mass.

    The same categorical legal choice one-token logits, AND the same
    observation-only additive prior, as the exact current model behavior.
    The gradient never targets tool-observation text or extra vocabulary.
    """
    import torch

    positives=sample.get("train_only_read_positive_indices")
    ids=sample.get("legal_choice_token_ids")
    actions=sample.get("legal_actions")
    prompt=sample.get("prompt_ids")
    biases=sample.get("action_logit_biases")
    if (not isinstance(positives,list) or not positives
        or not isinstance(ids,list) or not isinstance(actions,list)
        or len(ids)!=len(actions) or not ids
        or len(set(ids))!=len(ids)
        or len(set(positives))!=len(positives)
        or any(not isinstance(j,int) or isinstance(j,bool)
               or not 0<=j<len(ids) for j in positives)):
        raise ValueError("malformed TRAIN READ legal-target indices")
    if any(actions[j].get("kind")!="read" or
           not isinstance(actions[j].get("doc_id"),str)
           or not actions[j]["doc_id"] for j in positives):
        raise ValueError("TRAIN READ targets must be real legal READs")
    if (sample.get("action_token_targets")!=1
        or sample.get("tool_observation_targets")!=0):
        raise ValueError("read preference label masking violation")
    if not isinstance(prompt,list) or not prompt:
        raise ValueError("no real prompt for READ preference")
    tok=torch.tensor([prompt],device=model.device,dtype=torch.long)
    logits=model(input_ids=tok).logits[0,-1,:].float()
    subset=logits[torch.tensor(ids,device=logits.device)]
    if biases is not None:
        if (not isinstance(biases,list) or len(biases)!=len(ids)
            or any(not isinstance(x,(float,int)) or not math.isfinite(x)
                   or abs(x)>4 for x in biases)):
            raise ValueError("invalid behavior observation-only prior")
        subset=subset+torch.tensor(biases,device=logits.device,dtype=subset.dtype)
    lp=torch.log_softmax(subset,dim=-1)
    return -torch.logsumexp(lp[positives],dim=0)
