"""M7-A8: only authentic TRAIN original-support READs can be supervised."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from sidra.read_preference import annotate_train_read_preferences,read_positive_nll
from sidra.schema import Question


def _q(split="train"):
    return Question(
        question_id="train-mason",
        question="Which city is home to Mason Bridge?",
        gold_answer="Port",
        supporting_doc_ids=("a","b"),split=split,
    )


def _sample(choices,selected):
    return {
        "prompt_ids":[1,2,3],"legal_choice_token_ids":list(range(10,10+len(choices))),
        "action_logit_biases":[0.0]*len(choices),
        "selected_index":selected,"selected_token_id":10+selected,
        "legal_actions":choices,
        "selection_mode":"sample","tool_observation_targets":0,
        "action_token_targets":1,"sampled_logprob":-1.2,
    }


def _episode(*,seed=0,missing_hit=False,novel_read=False):
    search={"kind":"search","query":"Mason Bridge"}
    a={"kind":"read","doc_id":"a"}
    b={"kind":"read","doc_id":"b"}
    c={"kind":"read","doc_id":"c"}
    answer={"kind":"finish","answer":"Port","citations":("a","b")}
    hits=[{"doc_id":"a"},{"doc_id":"c"}] if missing_hit else [
        {"doc_id":"a"},{"doc_id":"b"},{"doc_id":"c"}]
    samples=[_sample([search],0),
             _sample([a,b,c],2),_sample([a,b],0),
             _sample([b],0),_sample([answer],0)]
    if novel_read:
        samples[2]=_sample([{"kind":"read","doc_id":"future"}],0)
    steps=[
        {"action":search,"observation":{"kind":"search","hits":hits}},
        {"action":c,"observation":{"kind":"read","doc_id":"c","text":"Irrelevant"}},
        {"action":samples[2]["legal_actions"][0],
         "observation":{"kind":"read","doc_id":"a","text":"Mason in Port"}},
        {"action":b,"observation":{"kind":"read","doc_id":"b","text":"Port city"}},
        {"action":answer},
    ]
    return {
        "qid":_q().question_id,
        "seed":seed,
        "trace":{"question_id":_q().question_id,"question":_q().question,
                 "steps":steps},
        "action_samples":samples,
    }


def test_existing_seen_train_support_read_gets_only_posthoc_labels():
    a=_episode(seed=5);b=_episode(seed=6)
    result=annotate_train_read_preferences([a,b],_q())
    assert result["eligible_real_read_states"]==6
    assert result["student_ignored_offered_support_read"]==2
    assert result["student_selected_support_read"]==4
    for ep in [a,b]:
        labels=[x.get("train_only_read_positive_indices")
                for x in ep["action_samples"]]
        assert labels==[None,[0,1],[0,1],[0],None]
        assert "gold_answer" not in json.dumps(ep["action_samples"][0])
        assert ep["action_samples"][1]["tool_observation_targets"]==0
        assert result["created_any_actions"] is False
    with pytest.raises(ValueError,match="overwrite"):
        annotate_train_read_preferences([a,b],_q())


def test_actual_search_hit_absence_blocks_future_gold_read_label():
    # Document b wasn't present in any prior real search hit, even though an
    # inconsistent artificial action lists it. Posthoc verifier must NOT label b.
    eps=[_episode(missing_hit=True),_episode(missing_hit=True)]
    result=annotate_train_read_preferences(eps,_q())
    assert eps[0]["action_samples"][1]["train_only_read_positive_indices"]==[0]
    assert eps[0]["action_samples"][2]["train_only_read_positive_indices"]==[0]
    assert "train_only_read_positive_indices" not in eps[0]["action_samples"][3]


def test_dev_gold_rejected_and_future_tool_read_invalid():
    with pytest.raises(ValueError,match="TRAIN"):
        annotate_train_read_preferences([_episode(),_episode()],_q("dev"))
    with pytest.raises(ValueError,match="real SEARCH"):
        annotate_train_read_preferences(
            [_episode(novel_read=True),_episode(novel_read=True)],_q())


def test_wrong_question_tampered_action_and_greedy_rejected():
    a=_episode();b=_episode()
    a["qid"]="dev-q"
    with pytest.raises(ValueError,match="identity"):
        annotate_train_read_preferences([a,b],_q())
    a=_episode();b=_episode()
    a["trace"]["steps"][1]["action"]={"kind":"read","doc_id":"ghost"}
    with pytest.raises(ValueError,match="authentic"):
        annotate_train_read_preferences([a,b],_q())
    a=_episode();b=_episode()
    a["action_samples"][1]["selection_mode"]="greedy"
    with pytest.raises(ValueError,match="authentic"):
        annotate_train_read_preferences([a,b],_q())


def test_masked_positive_read_nll_uses_only_legal_ids_and_saved_bias():
    torch=pytest.importorskip("torch")
    class Toy:
        def __init__(self):
            self.device=torch.device("cpu")
            self.logits=torch.nn.Parameter(
                torch.tensor([0.3,-0.2,0.5,-0.4,0.1]))
        def __call__(self,input_ids):
            n=input_ids.shape[-1]
            return SimpleNamespace(logits=self.logits.view(1,1,-1).expand(1,n,-1))
    m=Toy()
    action={
        "prompt_ids":[0,1],"legal_choice_token_ids":[1,3],
        "legal_actions":[{"kind":"read","doc_id":"b"},{"kind":"search","query":"bridge"}],
        "train_only_read_positive_indices":[0],
        "action_logit_biases":[0.4,-0.1],
        "action_token_targets":1,"tool_observation_targets":0,
    }
    loss=read_positive_nll(m,action)
    expected=torch.logsumexp(torch.tensor([-0.2+0.4,-0.4-0.1]),0)-0.2
    assert loss.item()==pytest.approx(expected.item(),rel=1e-5)
    loss.backward()
    assert m.logits.grad[[1,3]].abs().sum().item()>0
    assert m.logits.grad[[0,2,4]].abs().sum().item()==0


def test_masked_read_rejects_nonread_targets_or_invalid_token_mask():
    torch=pytest.importorskip("torch")
    class Toy:
        device=torch.device("cpu")
        def __call__(self,input_ids):
            return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[-1],20))
    action={
        "prompt_ids":[1],"legal_choice_token_ids":[10,11],
        "legal_actions":[{"kind":"search","query":"bridge"},
                         {"kind":"read","doc_id":"b"}],
        "train_only_read_positive_indices":[0],
        "action_logit_biases":[0.,0.],
        "action_token_targets":1,"tool_observation_targets":0,
    }
    with pytest.raises(ValueError,match="legal READ"):
        read_positive_nll(Toy(),action)
    action["train_only_read_positive_indices"]=[1]
    action["tool_observation_targets"]=1
    with pytest.raises(ValueError,match="masking"):
        read_positive_nll(Toy(),action)
    action["tool_observation_targets"]=0
    action["train_only_read_positive_indices"]=[1,1]
    with pytest.raises(ValueError,match="malformed"):
        read_positive_nll(Toy(),action)
    action["train_only_read_positive_indices"]=[1]
    action["action_logit_biases"]=[9.0,0.0]
    with pytest.raises(ValueError,match="prior"):
        read_positive_nll(Toy(),action)


def test_read_labels_survive_jsonl_round_trip():
    eps=json.loads(json.dumps([_episode(),_episode()]))
    audit=annotate_train_read_preferences(eps,_q())
    assert audit["eligible_real_read_states"]==6
