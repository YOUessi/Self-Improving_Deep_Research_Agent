"""Failure taxonomy distinguishes evidence, candidate and policy-choice bottlenecks."""
import copy
import pytest

from sidra.schema import Question
from scripts.audit_train_finish_bottlenecks import classify_train_episode


def _question(split="train"):
    return Question("train_X","Who chaired the original Atlas meeting?",
                    "Orion",("X::1","X::2"),split=split)


def _action(kind,answer="",doc_id="",citations=()):
    return {"kind":kind,"answer":answer,"doc_id":doc_id,
            "query":"Atlas" if kind=="search" else "",
            "citations":list(citations)}


def _pair(actions,idx=0,observation=None):
    picked=copy.deepcopy(actions[idx])
    step={"action":picked,"step":0}
    if observation is not None:step["observation"]=observation
    return step,{"legal_actions":copy.deepcopy(actions),"selected_index":idx,
                 "selection_mode":"sample","action_token_targets":1,
                 "tool_observation_targets":0}


def _build(q, *,missing_doc=False,offer_correct=True,
           select_correct=True,strict=True):
    gold=(_action("finish","Orion",citations=("X::1","X::2")))
    bad=_action("finish","Not Orion",citations=("X::1","X::2"))
    steps=[]
    samples=[]
    seq=[
        ([_action("search")],0,{"kind":"search","hits":[{"doc_id":"X::1"},{"doc_id":"X::2"}]}),
        ([_action("read",doc_id="X::1")],0,{"kind":"read","doc_id":"X::1"}),
    ]
    if not missing_doc:
        seq.append(([_action("read",doc_id="X::2")],0,
                    {"kind":"read","doc_id":"X::2"}))
    choices=([bad,gold] if offer_correct else [bad])
    ix=(1 if select_correct and offer_correct else 0)
    seq.append((choices,ix,None))
    for actions,idx,obs in seq:
        s,a=_pair(actions,idx,obs)
        steps.append(s);samples.append(a)
    return {"qid":q.question_id,
            "trace":{"question":q.question,"question_id":q.question_id,
                     "steps":steps},
            "train_reward":{"answer_correct":float(select_correct and offer_correct),
                            "strict_outcome":float(strict and select_correct
                                                    and offer_correct and not missing_doc)},
            "action_samples":samples}


def test_taxonomy_separates_evidence_catalog_and_choice():
    q=_question()
    assert classify_train_episode(_build(q),q)["reason"]=="strict_verified_success"
    assert classify_train_episode(
        _build(q,missing_doc=True,strict=False),q)["reason"]==(
            "missing_actually_read_original_gold_support")
    assert classify_train_episode(
        _build(q,offer_correct=False,strict=False),q)["reason"]==(
            "correct_answer_missing_from_action_catalog")
    r=classify_train_episode(
        _build(q,select_correct=False,strict=False),q)
    assert r["reason"]=="correct_fully_cited_finish_available_but_not_selected"
    assert r["support_fully_read"] and r["complete_correct_finish_offered"]
    assert classify_train_episode(
        _build(q,strict=False),q)["reason"]==(
            "correct_answer_selected_but_strict_verifier_rejected")


def test_requires_real_onpolicy_actions_train_labels_and_alignment():
    q=_question()
    with pytest.raises(ValueError,match="TRAIN"):
        classify_train_episode(_build(_question("dev")),_question("dev"))
    ep=_build(q)
    ep["qid"]="not_train_q"
    with pytest.raises(ValueError,match="mismatch"):
        classify_train_episode(ep,q)
    ep=_build(q)
    ep["action_samples"][0]["selection_mode"]="greedy"
    with pytest.raises(ValueError,match="mismatched or off-policy"):
        classify_train_episode(ep,q)
    ep=_build(q)
    ep["trace"]["steps"][0]["action"]["query"]="different"
    with pytest.raises(ValueError,match="mismatched or off-policy"):
        classify_train_episode(ep,q)
    ep=_build(q)
    ep["train_reward"]["strict_outcome"]=1.
    ep["action_samples"][-1]["selected_index"]=0
    ep["trace"]["steps"][-1]["action"]=_action(
        "finish","Not Orion",citations=("X::1","X::2"))
    with pytest.raises(ValueError,match="claimed strict success"):
        classify_train_episode(ep,q)
