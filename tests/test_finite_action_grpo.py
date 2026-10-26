"""Exact categorical constrained-action GRPO: safety and probability regression."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sidra.finite_action import (
    FiniteActionConfig,FiniteActionPolicy,action_prompt,build_legal_actions,
    choice_token_ids,logprob_of_action,
)
from sidra.schema import Question
from scripts.train_finite_action_grpo import select_train_only,finite_action_update


class TinyTokenizer:
    def encode(self,x,add_special_tokens=False):
        assert x.isdigit() and len(x)==1
        return [int(x)+2]

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,
                            enable_thinking=False):
        return "assistant chooses one from: "+messages[-1]["content"]

    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[2,3,4,5]}


class TinyModel:
    def __init__(self,torch):
        self.device=torch.device("cpu")
        self.weight=torch.nn.Parameter(torch.tensor(0.5))
        self.torch=torch

    def __call__(self,input_ids):
        torch=self.torch
        base=torch.arange(16,dtype=self.weight.dtype)
        # The output is a vocabulary score for the NEXT action ID token.
        logits=(base*self.weight).view(1,1,-1).expand(1,input_ids.shape[1],16)
        return SimpleNamespace(logits=logits)

    def eval(self):
        return self

    def parameters(self):
        return [self.weight]


def _toy_objects():
    torch=pytest.importorskip("torch")
    return torch,TinyModel(torch),TinyTokenizer()


def test_context_free_no_label_leak_and_legal_tools():
    obs=[{"kind":"search","query":"Who built Mason Bridge?",
          "hits":[{"doc_id":"q::a","title":"Mason Bridge","snippet":"Built by Riverhaven"}]},
         {"kind":"read","doc_id":"q::a","title":"Mason Bridge",
          "text":"Mason Bridge was designed by Riverhaven and opened in 1985."}]
    actions=build_legal_actions("Who built Mason Bridge?",obs)
    assert actions
    assert len(actions)<=9
    assert any(a.kind=="search" and "Riverhaven" in a.query for a in actions)
    assert any(a.kind=="finish" and a.answer for a in actions)
    assert all(a.kind!="read" for a in actions) # all exposed docs already read
    assert all(not a.citations or set(a.citations)=={"q::a"} for a in actions)
    assert any(a.kind=="finish" and not a.answer for a in actions)
    assert not any(getattr(a,"gold_answer",None) for a in actions)


def test_read_only_after_real_search_and_budgeted_catalog():
    a=build_legal_actions("Where did Lincoln grow up?",[])
    assert all(x.kind!="read" for x in a)
    obs=[{"kind":"search","query":"Where did Lincoln grow up?",
          "hits":[{"doc_id":str(i),"title":f"City {i}","snippet":"A city"}
                  for i in range(24)]}]
    acts=build_legal_actions("Where did Lincoln grow up?",obs)
    assert len(acts)<=9
    assert all(x.kind!="read" or x.doc_id in {"0","1","2"} for x in acts)
    assert not any(x.citations for x in acts)


def test_train_split_membership_and_dev_overlap():
    qs=[Question(str(i),"Q","gold",split="train") for i in range(10)]
    a=select_train_only(qs,set(),seed=4,n=4)
    b=select_train_only(list(reversed(qs)),set(),seed=4,n=4)
    assert [q.question_id for q in a]==[q.question_id for q in b]
    with pytest.raises(ValueError,match="overlaps"):
        select_train_only(qs,{"1"},seed=4,n=4)
    with pytest.raises(ValueError,match="TRAIN-only"):
        select_train_only([Question("d","Q","gold",split="dev")],set(),
                          seed=4,n=1)


def test_action_logprob_is_masked_normalizer_with_gradient():
    torch,m,tok=_toy_objects()
    legal=choice_token_ids(tok,3)
    assert legal==[2,3,4]
    lp=logprob_of_action(m,[1,1,1,1],legal,2)
    expected=torch.log_softmax(
        torch.tensor([2.,3.,4.])*m.weight,dim=0)[2]
    assert torch.allclose(lp,expected,atol=1e-6)
    assert lp.requires_grad
    lp.backward()
    assert torch.isfinite(m.weight.grad)
    assert abs(float(m.weight.grad))>1e-4


def test_policy_sampling_matches_legal_normalizer_and_masks_tool_text():
    torch,m,tok=_toy_objects()
    policy=FiniteActionPolicy(m,tok,seed=109)
    question="Where is Bridge Alpha?"
    choice=policy.decide(question,[],6)
    sample=policy.samples[0]
    assert choice.to_dict() in sample["legal_actions"]
    assert sample["action_token_targets"]==1
    assert sample["tool_observation_targets"]==0
    actual=float(logprob_of_action(
        m,sample["prompt_ids"],sample["legal_choice_token_ids"],
        sample["selected_index"]).detach())
    assert sample["sampled_logprob"]==pytest.approx(actual,abs=1e-6)
    assert math.exp(actual)==pytest.approx(sample["choice_prob"],abs=1e-6)


def test_optimizer_explicitly_rejects_off_policy_probabilities():
    torch,m,tok=_toy_objects()
    policy=FiniteActionPolicy(m,tok,seed=7)
    policy.decide("Who discovered Riverhaven?",[],5)
    sample=dict(policy.samples[0])
    sample["sampled_logprob"]+=1.5
    op=torch.optim.AdamW(m.parameters(),lr=1e-3)
    before=float(m.weight.detach())
    with pytest.raises(RuntimeError,match="distribution mismatch"):
        finite_action_update(m,[{"advantage":1.0,"action_samples":[sample]}],op)
    assert before==float(m.weight.detach())


def test_finite_action_gradients_update_only_selected_action_distribution():
    torch,m,tok=_toy_objects()
    policy=FiniteActionPolicy(m,tok,seed=3)
    policy.decide("What is the original name of the river?",[],5)
    sample=policy.samples[0]
    op=torch.optim.AdamW(m.parameters(),lr=1e-3)
    report=finite_action_update(
        m,[{"advantage":1.4,"action_samples":[sample]}],op)
    assert report["updated"] is True
    assert report["action_token_targets"]==1
    assert report["tool_text_token_targets"]==0
    assert report["behavior_replay_max_logprob_error"]<1e-5
    assert report["first_parameter_max_change"]>0


def test_invalid_action_id_tokenization_is_not_silently_allowed():
    class BadTokenizer(TinyTokenizer):
        def encode(self,x,add_special_tokens=False):
            return [1,2]
    with pytest.raises(ValueError,match="one tokenizer token"):
        choice_token_ids(BadTokenizer(),4)
