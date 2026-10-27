"""V5 finite catalog: exact one-token IDs, bounded research and no gold labels."""
import math
from types import SimpleNamespace

import pytest

from sidra.finite_action import (
    FiniteActionConfig,FiniteActionPolicy,build_legal_actions,
    choice_token_ids,logprob_of_action
)
from sidra.finish_candidates_v5 import visible_finish_options


READS=[
    {"kind":"search","query":"Was River City located on Island Green?",
     "hits":[{"doc_id":"a","title":"River City"},
             {"doc_id":"b","title":"Island Green"},
             {"doc_id":"c","title":"Elsewhere"}]},
    {"kind":"read","doc_id":"a","title":"River City",
     "text":"River City was built in 1957. John Mason founded River City."
            " The town was linked to Island Green. Thomas Edison visited it."},
    {"kind":"read","doc_id":"b","title":"Island Green",
     "text":"Island Green is a province on the coast. River City is in Island Green."
            " The island is governed by Northern Council. William Henry visited in 1954."},
]


class DigitTokenizer:
    def encode(self,text,add_special_tokens=False):
        assert len(text)==1 and text.isdigit()
        return [10+int(text)]

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,
                            enable_thinking=False):
        return "A:" + messages[-1]["content"]

    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[2,3,4]}


class TinyLM:
    def __init__(self,torch):
        self.device=torch.device("cpu")
        self.weight=torch.nn.Parameter(torch.tensor(0.51))
        self.torch=torch
    def __call__(self,input_ids):
        torch=self.torch
        scores=torch.arange(32,dtype=self.weight.dtype)*self.weight
        return SimpleNamespace(logits=scores.reshape(1,1,32).expand(
            1,input_ids.shape[-1],32))
    def eval(self):
        return self


def test_polar_questions_offer_both_truth_values_without_gold():
    question="Was River City located on Island Green?"
    cfg=FiniteActionConfig(answer_proposer="v5")
    allowed=build_legal_actions(question,READS,cfg=cfg,remaining=1,
                                max_tool_calls=5)
    finishes=[a for a in allowed if a.kind=="finish" and a.answer]
    answers=[a.answer for a in finishes]
    assert "yes" in answers and "no" in answers
    assert all(a.citations==("a","b") for a in finishes)
    assert len(allowed)<=10
    assert allowed[-1].kind=="finish" and not allowed[-1].answer
    assert not any(a.kind in {"read","search"} for a in allowed)
    assert all(not hasattr(a,"gold_answer") for a in allowed)
    old=build_legal_actions(question,READS,
             cfg=FiniteActionConfig(answer_proposer="v4"),
             remaining=1,max_tool_calls=5)
    assert all(a.answer not in {"yes","no"} for a in old if a.kind=="finish")


def test_v5_deploys_more_than_four_answers_at_final_step():
    text=("Mara was born in River City in 1950. Thomas Cole painted there "
          "in 1931. William Henry moved to Island Green in 1974. "
          "The town had Northern Council and Central Library buildings.")
    obs=READS.copy()
    obs[-1]={"kind":"read","doc_id":"b","title":"Island Green","text":text}
    question="Which artist moved to the island?"
    v4=build_legal_actions(question,obs,cfg=FiniteActionConfig(
        answer_proposer="v4"),remaining=1,max_tool_calls=5)
    v5=build_legal_actions(question,obs,cfg=FiniteActionConfig(
        answer_proposer="v5"),remaining=1,max_tool_calls=5)
    n4=sum(a.kind=="finish" and bool(a.answer) for a in v4)
    n5=sum(a.kind=="finish" and bool(a.answer) for a in v5)
    assert n4<=4
    assert n5>4
    assert n5<=8
    assert len(v5)<=10


def test_v5_keeps_real_read_and_search_while_bounded_to_single_digit():
    cfg=FiniteActionConfig(answer_proposer="v5")
    actions=build_legal_actions("Where is River City?",READS,
                               cfg=cfg,remaining=3,max_tool_calls=5)
    assert len(actions)<=10
    assert any(a.kind=="read" and a.doc_id=="c" for a in actions)
    assert any(a.kind=="search" for a in actions)
    assert actions[-1].kind=="finish" and actions[-1].answer==""
    assert len({(a.kind,a.query,a.doc_id,a.answer,a.citations) for a in actions})==len(actions)


def test_initial_and_one_read_v5_require_more_evidence():
    cfg=FiniteActionConfig(answer_proposer="v5")
    original=build_legal_actions("Where is River City?",[],
                                 cfg=cfg,remaining=6,max_tool_calls=5)
    assert not any(a.kind=="read" or (a.kind=="finish" and a.answer)
                   for a in original)
    one_read=build_legal_actions("Where is River City?",READS[:2],
                                 cfg=cfg,remaining=3,max_tool_calls=5)
    assert not any(a.kind=="finish" and a.answer for a in one_read)
    assert one_read[-1].kind=="finish" and one_read[-1].answer==""


def test_all_v5_decisions_have_exact_same_masked_normalizer_as_training():
    torch=pytest.importorskip("torch")
    model=TinyLM(torch)
    tok=DigitTokenizer()
    policy=FiniteActionPolicy(
        model,tok,seed=715,
        cfg=FiniteActionConfig(answer_proposer="v5"),
        max_tool_calls=5,selection_mode="sample")
    question="Was River City located on Island Green?"
    choice=policy.decide(question,READS,remaining=2)
    record=policy.samples[-1]
    assert choice.to_dict() in record["legal_actions"]
    assert 1<=record["num_legal_actions"]<=10
    assert all(len(tok.encode(str(i),add_special_tokens=False))==1
               for i in range(record["num_legal_actions"]))
    assert record["action_token_targets"]==1
    assert record["tool_observation_targets"]==0
    assert record["action_logit_biases"] is not None
    assert len(record["action_logit_biases"])==record["num_legal_actions"]
    logits=torch.tensor([
        float(logprob_of_action(
            model,record["prompt_ids"],record["legal_choice_token_ids"],
            i,logit_biases=record["action_logit_biases"]).detach())
        for i in range(record["num_legal_actions"])
    ])
    assert math.exp(float(record["sampled_logprob"]))==pytest.approx(
        float(record["choice_prob"]),abs=1e-6)
    assert float(torch.exp(logits).sum())==pytest.approx(1.0,abs=1e-6)
    picked=float(logits[record["selected_index"]])
    assert picked==pytest.approx(record["sampled_logprob"],abs=1e-6)
    assert record["selected_token_id"]==record["legal_choice_token_ids"][
        record["selected_index"]]


def test_v4_source_still_has_same_original_tool_candidates():
    q="When was River City built?"
    original=build_legal_actions(q,READS,
            cfg=FiniteActionConfig(answer_proposer="v4"),
            remaining=2,max_tool_calls=5)
    assert len(original)<=9
    assert all(a.kind=="read" and a.doc_id in {"c"} or
               a.kind in {"search","finish"} for a in original)
