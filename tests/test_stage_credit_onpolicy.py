"""Distinct stage credit (TRAIN-only reward after live tool interaction)."""
from types import SimpleNamespace

import pytest

from scripts.train_finite_action_grpo import stage_group_advantages,finite_action_update
from sidra.finite_action import FiniteActionPolicy


def _episode(read:float,correct:float,strict:float):
    return {"train_reward":{
        "read_support_recall":read,
        "answer_correct":correct,
        "strict_outcome":strict,
    }}


def test_stage_credits_reading_vs_answering_separately():
    # More relevant READ evidence but wrong FINISH should not be rewarded
    # as if it selected the right answer.
    a=_episode(read=1.,correct=0.,strict=0.)
    b=_episode(read=0.,correct=1.,strict=1.)
    stage_group_advantages([a,b])
    assert a["stage_advantages"]["read"]>0
    assert a["stage_advantages"]["finish"]<0
    assert b["stage_advantages"]["finish"]>0
    assert b["stage_advantages"]["search"]<0
    assert a["stage_advantages"]["read"]==a["stage_advantages"]["search"]


def test_constant_train_stage_reward_has_zero_credit():
    a=_episode(0.,0.,0.)
    b=_episode(0.,0.,0.)
    stage_group_advantages([a,b])
    assert a["stage_advantages"]=={"search":0.,"read":0.,"finish":0.}
    assert b["stage_advantages"]==a["stage_advantages"]
    with pytest.raises(ValueError,match=">=2"):
        stage_group_advantages([a])


class _Tokenizer:
    def encode(self,x,add_special_tokens=False):
        return [int(x)+2]
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,
                            enable_thinking=False):
        return "choose only a digit"
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[2,3,4,5]}


class _Model:
    def __init__(self,torch):
        self.device=torch.device("cpu")
        self.weight=torch.nn.Parameter(torch.tensor(0.5))
        self.torch=torch
    def __call__(self,input_ids):
        logits=(self.torch.arange(16,dtype=self.weight.dtype)*self.weight)
        return SimpleNamespace(logits=logits.view(1,1,-1).expand(
            1,input_ids.shape[1],16))
    def eval(self):return self
    def parameters(self):return [self.weight]


def test_stage_policy_update_is_true_live_masked_action_gradient():
    torch=pytest.importorskip("torch")
    m=_Model(torch)
    policy=FiniteActionPolicy(m,_Tokenizer(),seed=41)
    policy.decide("Who chaired the historical Vienna conference?",[],6)
    sample=policy.samples[0]
    kind=sample["legal_actions"][sample["selected_index"]]["kind"]
    reward={"search":0.,"read":0.,"finish":0.}
    reward[kind]=1.
    ep={"advantage":0.,
        "stage_advantages":reward,
        "action_samples":[sample]}
    op=torch.optim.AdamW(m.parameters(),lr=1e-3)
    report=finite_action_update(m,[ep],op,credit_mode="stage_credit_v1")
    assert report["updated"]
    assert report["credit_mode"]=="stage_credit_v1"
    assert report["behavior_replay_max_logprob_error"]<1e-5
    assert report["action_token_targets"]==1
    assert report["tool_text_token_targets"]==0


def test_stage_credit_requires_all_three_explicit_train_reward_roles():
    torch=pytest.importorskip("torch")
    m=_Model(torch)
    policy=FiniteActionPolicy(m,_Tokenizer(),seed=3)
    policy.decide("What bridge is this?",[],5)
    ep={"advantage":0.,"stage_advantages":{"read":1.},
        "action_samples":[policy.samples[0]]}
    op=torch.optim.AdamW(m.parameters(),lr=1e-3)
    with pytest.raises(ValueError,match="TRAIN stage advantages absent"):
        finite_action_update(m,[ep],op,credit_mode="stage_credit_v1")
    with pytest.raises(ValueError,match="credit mode"):
        finite_action_update(m,[ep],op,credit_mode="unsafe_oracle")
