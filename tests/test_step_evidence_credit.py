"""TRAIN-only per-decision search/read/finish reward: no model gold exposure."""
import copy
import pytest

from sidra.schema import Question
from sidra.step_credit import assign_step_evidence_credit


def _q(split="train"):
    return Question("train_42","Which city hosted the Atlas summit?",
                    "Harbor City",("train_42::a","train_42::b"),split=split)


def _s(kind,obs=None,doc_id="",answer=""):
    action={"kind":kind,"doc_id":doc_id,"query":"Atlas summit" if kind=="search" else "",
            "answer":answer,"citations":["train_42::a","train_42::b"]
               if kind=="finish" and answer else []}
    return ({"action":action,**({"observation":obs} if obs is not None else {})},
            {"legal_actions":[copy.deepcopy(action)],
             "selected_index":0,"action_token_targets":1,
             "tool_observation_targets":0,"selection_mode":"sample",
             "prompt_ids":[1,2,3],
             "legal_choice_token_ids":[4],"selected_token_id":4,
             "sampled_logprob":0.0})


def _episode(q,seq,strict=False,correct=False):
    steps,samples=[],[]
    for kind,obs,doc_id,answer in seq:
        step,sample=_s(kind,obs,doc_id,answer)
        steps.append(step);samples.append(sample)
    return {"qid":q.question_id,
            "trace":{"question_id":q.question_id,
                     "question":q.question,
                     "steps":steps,"termination":"finish"},
            "train_reward":{"answer_correct":float(correct),
                            "strict_outcome":float(strict)},
            "action_samples":samples}


def _search(*ids):
    return {"kind":"search","hits":[{"doc_id":v} for v in ids]}


def _read(id):
    return {"kind":"read","doc_id":id,"text":"example text", "title":"a"}


def examples(q=None):
    q=q or _q()
    good=_episode(q,[
        ("search",_search("train_42::a","train_42::b"),"",""),
        ("read",_read("train_42::a"),"train_42::a",""),
        ("read",_read("train_42::b"),"train_42::b",""),
        ("finish",None,"","Harbor City")],strict=True,correct=True)
    partial=_episode(q,[
        ("search",_search("train_42::a"),"",""),
        ("read",_read("train_42::a"),"train_42::a",""),
        ("read",_read("train_42::unrelated"),"train_42::unrelated",""),
        ("finish",None,"","Unknown")],strict=False,correct=False)
    bad=_episode(q,[
        ("search",_search("train_42::unrelated"),"",""),
        ("read",_read("train_42::unrelated"),"train_42::unrelated",""),
        ("finish",None,"","Unknown")],strict=False,correct=False)
    return [good,partial,bad]


def test_step_credit_is_actual_incremental_gold_discovery_not_terminal_leakage():
    q=_q()
    ep=examples(q)
    rep=assign_step_evidence_credit(ep,q)
    assert rep["gold_support_source"]=="original TRAIN only"
    assert not rep["policy_or_action_catalog_uses_gold"]
    assert rep["by_kind"]["search"]["positive_raw"]==2
    assert rep["by_kind"]["read"]["positive_raw"]==3
    assert rep["by_kind"]["finish"]["positive_raw"]==1
    assert rep["credited_actions"]>0
    assert rep["assistant_action_tokens"]==len(sum((e["action_samples"] for e in ep),[]))
    assert ep[0]["action_samples"][0]["step_credit_raw"]==1.
    assert ep[0]["action_samples"][1]["step_credit_raw"]==0.5
    assert ep[0]["action_samples"][2]["step_credit_raw"]==0.5
    assert ep[0]["action_samples"][3]["step_credit_raw"]==1.
    assert ep[1]["action_samples"][2]["step_credit_raw"]==0.
    # Every action gets a separate optimization credit: no broadcast of a
    # winning episode's READ score to an unrelated late READ.
    assert ep[0]["action_samples"][2]["step_credit_advantage"]>0
    assert ep[1]["action_samples"][2]["step_credit_advantage"]<0
    # Crucial: raw model input and legal catalog have not been modified.
    for e in ep:
        for s in e["action_samples"]:
            assert s["prompt_ids"]==[1,2,3]
            assert set(s["legal_actions"][0]).isdisjoint({"gold_answer","supporting_doc_ids"})


def test_repeated_gold_hits_do_not_inflate_search_credit():
    q=_q()
    a=_episode(q,[
        ("search",_search("train_42::a"),"",""),
        ("search",_search("train_42::a"),"",""),
        ("read",_read("train_42::a"),"train_42::a",""),
        ("finish",None,"","Unknown")])
    b=copy.deepcopy(a)
    rep=assign_step_evidence_credit([a,b],q)
    assert a["action_samples"][0]["step_credit_raw"]==0.5
    assert a["action_samples"][1]["step_credit_raw"]==0.
    assert rep["credited_actions"]==0, "constant group can't produce fictitious gradients"


def test_guards_reject_nontrain_and_cross_question_or_missing_tool_observations():
    q=_q()
    with pytest.raises(ValueError,match="TRAIN"):
        assign_step_evidence_credit(examples(_q("dev")),_q("dev"))
    eps=examples(q)
    eps[1]["qid"]="different"
    with pytest.raises(ValueError,match="ID mismatch"):
        assign_step_evidence_credit(eps,q)
    eps=examples(q)
    eps[0]["trace"]["steps"][0].pop("observation")
    with pytest.raises(ValueError,match="really executed"):
        assign_step_evidence_credit(eps,q)
    eps=examples(q)
    eps[0]["action_samples"][1]["legal_actions"][0]["doc_id"]="forged"
    with pytest.raises(ValueError,match="mismatch"):
        assign_step_evidence_credit(eps,q)
    eps=examples(q)
    eps[0]["action_samples"][1]["tool_observation_targets"]=1
    with pytest.raises(ValueError,match="mismatch"):
        assign_step_evidence_credit(eps,q)
    with pytest.raises(ValueError,match="original TRAIN supporting IDs"):
        assign_step_evidence_credit(
            examples(q),
            Question(q.question_id,q.question,q.gold_answer,(),split="train"))


def test_inconsistent_terminal_reward_is_rejected():
    eps=examples()
    eps[0]["train_reward"]={"answer_correct":0.0,"strict_outcome":1.0}
    with pytest.raises(ValueError,match="strict success"):
        assign_step_evidence_credit(eps,_q())
