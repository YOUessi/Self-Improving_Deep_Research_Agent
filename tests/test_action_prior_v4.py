"""M7-A6 v4 exact probability replay and trust-boundary regressions.

These tests do not claim QA generalization. v4 is an opt-in ablation and does
not alter v3's original categorical distribution.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sidra.action_prior import action_logit_biases
from sidra.finite_action import (
    FiniteActionConfig,FiniteActionPolicy,build_legal_actions,logprob_of_action
)
from sidra.schema import Action
from scripts.train_finite_action_grpo import finite_action_update


QUESTION="Which railway connects Sterling, Montana to Red Lake?"
OBS=[
    {"kind":"search","query":QUESTION,"hits":[
        {"doc_id":"q::first","title":"Sterling Montana",
         "snippet":"Sterling was built on a major railroad"},
        {"doc_id":"q::linked","title":"Northern Pacific Railroad",
         "snippet":"Northern Pacific Railroad near Sterling Montana"},
        {"doc_id":"q::unrelated","title":"Tennis rankings",
         "snippet":"Tennis ranking sport players"},
    ]},
    {"kind":"read","doc_id":"q::first","title":"Sterling Montana",
     "text":"Sterling, Montana was established beside the Northern Pacific Railroad. "
            "The line continued to Red Lake, where the depot was expanded."},
]


class ToyTokenizer:
    def encode(self,word,add_special_tokens=False):
        return [2+int(word)]
    def apply_chat_template(self,messages,**kwargs):
        return "valid:" + messages[-1]["content"]
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[1,2,3,4,5]}


class ToyModel:
    def __init__(self,torch):
        self.torch=torch
        self.device=torch.device("cpu")
        self.weight=torch.nn.Parameter(torch.tensor(0.2))
    def __call__(self,input_ids):
        torch=self.torch
        logits=(torch.arange(32)*self.weight).reshape(1,1,-1)
        return SimpleNamespace(logits=logits)
    def eval(self):
        return self
    def parameters(self):
        return [self.weight]


def test_v4_action_prior_boosts_observed_bridge_read_without_gold():
    actions=[
        Action(kind="read",doc_id="q::linked"),
        Action(kind="read",doc_id="q::unrelated"),
        Action(kind="search",query="Red Lake"),
        Action(kind="finish",answer="",citations=()),
    ]
    original=action_logit_biases(QUESTION,OBS,actions)
    assert len(original)==4
    assert original[0]>original[1]
    assert original[3]<original[2]
    # These fields are evaluator-only contaminants, not an observation.
    # The *computed action prior* cannot use them.
    contaminated=[{**o,"gold_answer":"Tennis rankings",
                     "supporting_doc_ids":["q::unrelated"]} for o in OBS]
    assert action_logit_biases(QUESTION,contaminated,actions)==original
    assert max(map(abs,original))<=4


def test_v4_respects_observed_only_legal_read_and_finish_and_budget():
    cfg=FiniteActionConfig(answer_proposer="v4")
    candidates=build_legal_actions(QUESTION,OBS,cfg=cfg,remaining=4,max_tool_calls=5)
    assert len(candidates)<=9
    assert any(x.kind=="read" and x.doc_id=="q::linked" for x in candidates)
    assert not any(x.kind=="read" and x.doc_id=="unseen" for x in candidates)
    assert not any(x.kind=="finish" and x.answer for x in candidates), (
        "single read is not sufficient for a two-hop factual answer")
    assert any(x.kind=="search" for x in candidates)
    final=build_legal_actions(QUESTION,OBS,cfg=cfg,remaining=1,max_tool_calls=5)
    assert len(final)==1 and final[0].kind=="finish" and not final[0].answer


def test_v4_catalog_exposes_read_bridge_query_from_real_observation():
    cfg=FiniteActionConfig(answer_proposer="v4")
    original=build_legal_actions(QUESTION,OBS,cfg=cfg,remaining=4)
    queries=[a.query for a in original if a.kind=="search"]
    assert queries
    assert len(queries)==len(set(queries))
    # No TRAIN/DEV label text can enter from non-observation keys.
    contaminated=[{**x,"gold_answer":"MAGIC_GOLD_NEVER_SEARCH",
                    "supporting_doc_ids":["MAGIC_GOLD_NEVER_SEARCH"]}
                  for x in OBS]
    assert build_legal_actions(QUESTION,contaminated,cfg=cfg,
                               remaining=4)==original


def test_v4_selected_action_probability_exact_replay_and_action_only_mask():
    torch=pytest.importorskip("torch")
    model=ToyModel(torch)
    policy=FiniteActionPolicy(model,ToyTokenizer(),seed=93,
                              cfg=FiniteActionConfig(answer_proposer="v4"))
    chosen=policy.decide(QUESTION,OBS,remaining=4)
    sample=policy.samples[-1]
    biases=sample["action_logit_biases"]
    assert biases is not None and len(biases)==len(sample["legal_actions"])
    assert sample["legal_actions"][sample["selected_index"]]==chosen.to_dict()
    assert sample["action_token_targets"]==1
    assert sample["tool_observation_targets"]==0
    expected=float(logprob_of_action(
        model,sample["prompt_ids"],sample["legal_choice_token_ids"],
        sample["selected_index"],logit_biases=biases).detach())
    assert sample["sampled_logprob"]==pytest.approx(expected,abs=1e-6)
    assert math.exp(expected)==pytest.approx(sample["choice_prob"],rel=1e-5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    report=finite_action_update(
        model,[{"advantage":1.0,"action_samples":[sample]}],optimizer)
    assert report["updated"]
    assert report["behavior_replay_max_logprob_error"]<1e-5
    assert report["action_token_targets"]==1
    assert report["tool_text_token_targets"]==0


def test_v4_tampered_bias_refused_before_any_optimizer_step():
    torch=pytest.importorskip("torch")
    model=ToyModel(torch)
    policy=FiniteActionPolicy(model,ToyTokenizer(),seed=93,
                              cfg=FiniteActionConfig(answer_proposer="v4"))
    policy.decide(QUESTION,OBS,remaining=4)
    sample=dict(policy.samples[-1])
    sample["action_logit_biases"]=[x + (3.0 if i==0 else 0.0)
                                  for i,x in enumerate(sample["action_logit_biases"])]
    before=float(model.weight.detach())
    with pytest.raises(RuntimeError,match="distribution mismatch"):
        finite_action_update(model,[{"advantage":1.0,
                                    "action_samples":[sample]}],
                             torch.optim.AdamW(model.parameters(),lr=1e-3))
    assert before==float(model.weight.detach())


def test_prior_refuses_illegal_catalog_and_nonfinite_logit_prior():
    with pytest.raises(ValueError,match="unseen"):
        action_logit_biases(QUESTION,OBS,[Action(kind="read",doc_id="fake")])
    torch=pytest.importorskip("torch")
    model=ToyModel(torch)
    with pytest.raises(ValueError,match="prior"):
        logprob_of_action(model,[1,2],[2,3],0,logit_biases=[0.4])
    with pytest.raises(ValueError,match="prior"):
        logprob_of_action(model,[1,2],[2,3],0,
                          logit_biases=[float("nan"),0])
