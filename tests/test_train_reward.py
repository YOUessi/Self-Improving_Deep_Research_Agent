"""Train-only multi-hop reward must not reward empty shortcuts or leak dev gold."""
import pytest

from sidra.schema import Action, AgentTrace, Document, Question
from sidra.train_reward import score_train_rollout,group_normalized_advantages


DOCS=[
    Document("person","Person","Person lives in Port"),
    Document("region","Region","Port belongs to North"),
]
LOOKUP={d.doc_id:d for d in DOCS}


def make_rollout(*, answer="North", citations=("person","region"),
                 read_ids=("person","region"), termination="finish",calls=4):
    steps=[{"action":Action("search",query="Person Port").to_dict(),
            "observation":{"kind":"search","query":"Person Port",
                "hits":[{"doc_id":"person"},{"doc_id":"region"}]}}]
    for doc_id in read_ids:
        steps.append({"action":Action("read",doc_id=doc_id).to_dict(),
                      "observation":{"kind":"read","doc_id":doc_id,
                                     "text":LOOKUP[doc_id].text}})
    if termination=="finish":
        steps.append({"action":Action("finish",answer=answer,
                      citations=tuple(citations)).to_dict()})
    return AgentTrace(question_id="q",question="Where?",policy="test",
                      answer=answer,citations=list(citations),
                      termination=termination,steps=steps,tool_calls=calls)


def train_question(split="train"):
    return Question("q","Where?","North",
                    supporting_doc_ids=("person","region"),split=split)


def test_verified_answer_plus_evidence_dominates_reward():
    good=score_train_rollout(train_question(),make_rollout(),LOOKUP)
    assert good.strict_outcome==1 and good.cited_support_complete==1
    assert good.answer_correct==1
    assert 0.9<=good.total<=1
    wrong=score_train_rollout(train_question(),
            make_rollout(answer="Other"),LOOKUP)
    assert wrong.strict_outcome==0
    assert wrong.total<good.total
    unsupported=score_train_rollout(train_question(),
            make_rollout(citations=("person",),read_ids=("person",)),
            LOOKUP)
    assert unsupported.strict_outcome==0
    assert unsupported.total<good.total


def test_cheap_abstain_and_duplicate_search_cannot_win():
    q=train_question()
    abstain=score_train_rollout(q,make_rollout(answer="",citations=(),
                     read_ids=(),calls=0),LOOKUP)
    assert abstain.total<=0.05
    assert abstain.efficiency_bonus==0
    too_many=score_train_rollout(q,make_rollout(calls=5),LOOKUP)
    assert too_many.total<=score_train_rollout(q,make_rollout(calls=3),LOOKUP).total


def test_dev_test_rewards_are_blocked():
    for split in ("dev","test"):
        with pytest.raises(ValueError,match="TRAIN-only"):
            score_train_rollout(train_question(split),make_rollout(),LOOKUP)


def test_group_advantages_have_zero_mean_and_collapse_when_constant():
    a=group_normalized_advantages([0.1,0.2,0.9,0.8])
    assert abs(sum(a))<1e-9
    assert a[2]>a[1]
    assert group_normalized_advantages([0.3,0.3,0.3,0.3])==[0,0,0,0]
    with pytest.raises(ValueError):group_normalized_advantages([])
