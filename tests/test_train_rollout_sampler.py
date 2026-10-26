"""RL preflight samples TRAIN-only questions and distinguishes rollout from update."""
import pytest
from sidra.schema import Question
from scripts.sample_train_rollouts import select_train_questions,summarize_groups


def questions(split="train"):
    return [Question(f"q{i}",f"what {i}?","answer",split=split) for i in range(20)]


def test_train_only_selection_frozen_and_dev_rejected():
    qs=questions()
    a=select_train_questions(qs,{"dev-other"},limit=8,seed=43)
    b=select_train_questions(list(reversed(qs)),{"dev-other"},limit=8,seed=43)
    assert [x.question_id for x in a]==[x.question_id for x in b]
    with pytest.raises(ValueError):
        select_train_questions(qs,{"q3"},limit=8,seed=43)
    with pytest.raises(ValueError):
        select_train_questions(questions("dev"),set(),limit=8,seed=43)


def test_reward_spread_accounts_for_actual_group_variation():
    def row(reward,success,terminal):
        return {"reward":{"total":reward,"strict_outcome":success,
                          "read_support_recall":0.5},
                "trace":{"termination":terminal}}
    g=[
        [row(.1,0,"error"),row(.1,0,"error")],
        [row(.05,0,"abstained"),row(.85,1,"finish")],
    ]
    s=summarize_groups(g)
    assert s["train_groups"]==2
    assert s["variable_reward_groups"]==1
    assert s["variable_reward_group_fraction"]==0.5
    assert s["strictly_correct_training_rollouts"]==1
    assert s["finish_reasons"]["finish"]==1
    with pytest.raises(ValueError):summarize_groups([])
