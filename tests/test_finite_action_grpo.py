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


def test_evaluator_manifest_model_name_is_json_serializable(tmp_path):
    import json
    from sidra.agent import AgentConfig
    from sidra.experiment import evaluate
    from sidra.io import save_jsonl
    torch,m,tok=_toy_objects()
    data=tmp_path/"source"
    data.mkdir()
    save_jsonl(data/"documents.jsonl",[
        {"doc_id":"d1","title":"Mason Bridge","text":"Mason Bridge is in Riverhaven"}])
    save_jsonl(data/"questions.jsonl",[
        {"question_id":"q1","question":"Where is Mason Bridge?",
         "gold_answer":"Riverhaven","split":"dev",
         "supporting_doc_ids":["d1"]}])
    policy=FiniteActionPolicy(
        m,tok,seed=3,model_name="toy/fake-base",adapter_path="toy/fake-lora")
    result=evaluate(
        data,"dev",policy,tmp_path/"report",
        config=AgentConfig(max_steps=3,max_tool_calls=2),limit=1)
    manifest=json.loads((tmp_path/"report"/"manifest.json").read_text())
    assert result["n"]==1
    assert manifest["model_id"]=="toy/fake-base"
    assert manifest["adapter_path"]=="toy/fake-lora"
    assert manifest["no_gpu_training_claimed"] is True


def test_terminal_action_catalog_excludes_tool_calls_on_last_step():
    obs=[{"kind":"search","query":"Find Bridge",
          "hits":[{"doc_id":"docA","title":"Bridge","snippet":"Bridge site"}]},
         {"kind":"read","doc_id":"docA","title":"Bridge",
          "text":"Bridge site was built by Riverhaven in 1985."}]
    result=build_legal_actions("When was the bridge built?",obs,
                               remaining=1,max_tool_calls=5)
    assert result
    assert all(a.kind=="finish" for a in result)
    assert any(a.answer for a in result)
    assert all(not a.citations or set(a.citations)=={"docA"} for a in result)
    no_read=build_legal_actions("What is it?",[],remaining=1)
    assert len(no_read)==1 and no_read[0].kind=="finish"
    assert no_read[0].answer==""
    with pytest.raises(ValueError,match="remaining"):
        build_legal_actions("What?",[],remaining=0)


def test_catalog_prevents_search_or_read_after_tool_budget_hit():
    obs=[{"kind":"search","query":"The bridge",
          "hits":[{"doc_id":"d","title":"Bridge","snippet":"Bridge"}]},
         {"kind":"read","doc_id":"d","title":"Bridge",
          "text":"Bridge was built in 1985 by Riverhaven."}]
    maxed=build_legal_actions("When was bridge built?",obs,
                              remaining=3,max_tool_calls=2)
    assert all(a.kind=="finish" for a in maxed)
    notmax=build_legal_actions("When was bridge built?",obs,
                               remaining=3,max_tool_calls=3)
    assert any(a.kind=="search" for a in notmax)
    with pytest.raises(ValueError,match="tool budget"):
        build_legal_actions("When?",obs,max_tool_calls=0)


def test_rng_is_keyed_by_local_state_not_previous_questions():
    torch,m,tok=_toy_objects()
    seq=FiniteActionPolicy(m,tok,seed=222,selection_mode="sample")
    initial=seq.decide("Where is Mason Bridge?",[],6).to_dict()
    first=seq.samples[-1]
    seq.decide("Which year was the bridge opened?",[],6)
    again=seq.decide("Where is Mason Bridge?",[],6).to_dict()
    last=seq.samples[-1]
    assert initial==again
    assert first["decision_seed"]==last["decision_seed"]
    assert first["selected_index"]==last["selected_index"]
    assert first["sampled_logprob"]==pytest.approx(last["sampled_logprob"])


def test_greedy_decisions_are_not_passed_off_as_on_policy_samples():
    torch,m,tok=_toy_objects()
    greedy=FiniteActionPolicy(m,tok,seed=34,selection_mode="greedy")
    greedy.decide("What city was he born in?",[],6)
    sample=greedy.samples[0]
    assert sample["selection_mode"]=="greedy"
    assert sample["selected_index"]==sample["num_legal_actions"]-1 or (
        sample["selected_index"]>=0)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    with pytest.raises(ValueError,match="off-policy"):
        finite_action_update(
            m,[{"advantage":1.0,"action_samples":[sample]}],opt)
    with pytest.raises(ValueError,match="selection mode"):
        FiniteActionPolicy(m,tok,selection_mode="invalid")


def test_v2_catalog_can_propose_unicode_multiword_person_from_read_evidence():
    from sidra.finite_action import FiniteActionConfig
    text=("The Congress of Vienna was chaired by Austrian statesman "
          "Klemens von Metternich, between November 1814 and June 1815.")
    observations=[
        {"kind":"search","query":"Austrian statesman Congress",
         "hits":[{"doc_id":"train::0","title":"Congress","snippet":"Austrian statesman"}]},
        {"kind":"read","doc_id":"train::0","title":"Congress","text":text},
    ]
    question="Which Austrian statesman chaired the meeting?"
    legacy=build_legal_actions(question,observations,remaining=1,
                               cfg=FiniteActionConfig(answer_proposer="legacy"))
    upgraded=build_legal_actions(question,observations,remaining=1,
                                 cfg=FiniteActionConfig(answer_proposer="v2"))
    assert not any(a.kind=="finish" and a.answer=="Klemens von Metternich"
                   for a in legacy)
    assert any(a.kind=="finish" and a.answer=="Klemens von Metternich"
               and a.citations==("train::0",) for a in upgraded)
    with pytest.raises(ValueError,match="answer proposer"):
        FiniteActionConfig(answer_proposer="gold_oracle")


def test_distribution_mismatch_in_late_action_is_rejected_before_any_update():
    torch,m,tok=_toy_objects()
    policy=FiniteActionPolicy(m,tok,seed=43)
    policy.decide("When was the bridge built?",[],5)
    record=policy.samples[0]
    actions=[dict(record) for _ in range(14)]
    actions[-1]["sampled_logprob"]+=0.3
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    before=float(m.weight.detach())
    with pytest.raises(RuntimeError,match="distribution mismatch"):
        finite_action_update(m,[{"advantage":1.0,
                                  "action_samples":actions}],opt)
    assert float(m.weight.detach())==before
