"""M7-A7 preference labels are TRAIN-only and never part of policy sampling."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from sidra.finish_preference import (
    annotate_train_finish_preferences,finish_positive_nll,
)
from sidra.schema import Question


def _q(split="train"):
    return Question(
        question_id="train-q",question="What city contains the Mason Bridge?",
        gold_answer="Port",supporting_doc_ids=("a","b"),split=split,
    )


def _sample(action,choices=None,selected=0):
    choices=choices or [action]
    return {
        "prompt_ids":[1,2,3],
        "legal_choice_token_ids":list(range(10,10+len(choices))),
        "action_logit_biases":[0.1]*len(choices),
        "selected_index":selected,
        "selected_token_id":10+selected,
        "selection_mode":"sample","tool_observation_targets":0,
        "action_token_targets":1,
        "legal_actions":choices,
        "sampled_logprob":-0.7,
    }


def _episode(*,missing_second_read=False,correct_answer_in_read=True,
             final_selected=1,bad_final_citations=False):
    search={"kind":"search","query":"Mason Bridge Port"}
    a={"kind":"read","doc_id":"a"}
    b={"kind":"read","doc_id":"b"}
    cite=["a","nonexistent"] if bad_final_citations else ["a","b"]
    correct={"kind":"finish","answer":"Port","citations":cite}
    wrong={"kind":"finish","answer":"An Unrelated City","citations":["a","b"]}
    steps=[
        {"action":search,"observation":{"kind":"search",
            "hits":[{"doc_id":"a"},{"doc_id":"b"}]}},
        {"action":a,"observation":{"kind":"read","doc_id":"a",
            "text":"Mason Bridge stands in Port." if correct_answer_in_read
                else "Mason Bridge stands in a coastal city."}},
    ]
    samples=[_sample(search),_sample(a)]
    if not missing_second_read:
        steps.append({"action":b,"observation":{"kind":"read","doc_id":"b",
                                               "text":"Another source confirms this bridge."}})
        samples.append(_sample(b))
    finish=[correct,wrong]
    steps.append({"action":finish[final_selected]})
    samples.append(_sample(finish[final_selected],finish,final_selected))
    return {
        "qid":"train-q","generation":0,"trace":{
            "question_id":"train-q","question":_q().question,
            "steps":steps,
        },"action_samples":samples,
    }


def test_labels_only_after_both_actual_reads_and_in_original_catalog():
    episodes=[_episode(),_episode(final_selected=0)]
    report=annotate_train_finish_preferences(episodes,_q())
    assert report["eligible_states"]==2
    assert report["student_ignored_correct_finish"]==1
    assert report["student_chose_correct_finish"]==1
    for ep in episodes:
        assert all("train_only_finish_positive_indices" not in s
                   for s in ep["action_samples"][:-1])
        assert ep["action_samples"][-1]["train_only_finish_positive_indices"]==[0]
        assert ep["action_samples"][-1]["action_token_targets"]==1
        assert ep["action_samples"][-1]["tool_observation_targets"]==0


def test_unread_support_cannot_be_relabelled_as_valid_finish():
    episodes=[_episode(missing_second_read=True) for _ in range(2)]
    report=annotate_train_finish_preferences(episodes,_q())
    assert report.get("eligible_states",0)==0
    assert not any("train_only_finish_positive_indices" in x
                   for ep in episodes for x in ep["action_samples"])


def test_late_or_unseen_gold_citation_is_not_valid_positive():
    episodes=[_episode(bad_final_citations=True) for _ in range(2)]
    report=annotate_train_finish_preferences(episodes,_q())
    assert report.get("eligible_states",0)==0


def test_answer_must_be_found_in_actually_read_cited_text():
    episodes=[_episode(correct_answer_in_read=False) for _ in range(2)]
    report=annotate_train_finish_preferences(episodes,_q())
    assert report.get("eligible_states",0)==0


def test_dev_labels_rejected_even_with_identical_task_structure():
    with pytest.raises(ValueError,match="original TRAIN"):
        annotate_train_finish_preferences([_episode(),_episode()],_q("dev"))


def test_tampered_action_trace_refused():
    episodes=[_episode(),_episode()]
    episodes[0]["trace"]["steps"][1]["action"]={"kind":"read","doc_id":"fake"}
    with pytest.raises(ValueError,match="authentic"):
        annotate_train_finish_preferences(episodes,_q())


def test_wrong_question_or_reannotation_fails_closed():
    eps=[_episode(),_episode()]
    eps[0]["qid"]="dev"
    with pytest.raises(ValueError,match="identity"):
        annotate_train_finish_preferences(eps,_q())
    eps=[_episode(),_episode()]
    annotate_train_finish_preferences(eps,_q())
    with pytest.raises(ValueError,match="overwrite"):
        annotate_train_finish_preferences(eps,_q())


def test_gradient_targets_only_legal_choice_ids_and_has_behavior_prior():
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
    sample={
        "prompt_ids":[0,1,2],"legal_choice_token_ids":[1,3],
        "legal_actions":[
            {"kind":"finish","answer":"Port"},
            {"kind":"finish","answer":"Other"}],
        "train_only_finish_positive_indices":[0],
        "action_logit_biases":[0.4,-0.1],
        "action_token_targets":1,"tool_observation_targets":0,
    }
    loss=finish_positive_nll(m,sample)
    expected=torch.logsumexp(torch.tensor([-0.2+0.4,-0.4-0.1]),0)-(0.2)
    assert loss.item()==pytest.approx(expected.item(),rel=1e-5)
    loss.backward()
    assert abs(m.logits.grad[1].item())>0
    assert abs(m.logits.grad[3].item())>0
    assert m.logits.grad[[0,2,4]].abs().sum().item()==0


def test_unverified_or_nonfinish_target_refused():
    torch=pytest.importorskip("torch")
    class Toy:
        device=torch.device("cpu")
        def __call__(self,input_ids):
            return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[-1],20))
    base={
        "prompt_ids":[1],"legal_choice_token_ids":[10,11],
        "legal_actions":[{"kind":"read","doc_id":"a"},
                         {"kind":"finish","answer":"Port"}],
        "train_only_finish_positive_indices":[0],
        "action_logit_biases":[0.,0.],
        "action_token_targets":1,"tool_observation_targets":0,
    }
    with pytest.raises(ValueError,match="factual legal FINISH"):
        finish_positive_nll(Toy(),base)
    d=copy.deepcopy(base);d["train_only_finish_positive_indices"]=[1]
    d["tool_observation_targets"]=2
    with pytest.raises(ValueError,match="masking"):
        finish_positive_nll(Toy(),d)


def test_out_of_bounds_bias_and_duplicate_positive_indices_rejected():
    torch=pytest.importorskip("torch")
    class Toy:
        device=torch.device("cpu")
        def __call__(self,input_ids):
            return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[-1],20))
    sample={
        "prompt_ids":[1],"legal_choice_token_ids":[10,11],
        "legal_actions":[{"kind":"finish","answer":"Port"},
                         {"kind":"finish","answer":"Port"}],
        "train_only_finish_positive_indices":[0,0],
        "action_logit_biases":[0.,0.],
        "action_token_targets":1,"tool_observation_targets":0,
    }
    with pytest.raises(ValueError,match="malformed"):
        finish_positive_nll(Toy(),sample)
    sample["train_only_finish_positive_indices"]=[0]
    sample["action_logit_biases"]=[5,0]
    with pytest.raises(ValueError,match="prior"):
        finish_positive_nll(Toy(),sample)
