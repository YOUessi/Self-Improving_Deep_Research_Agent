"""Qwen3 SFT sample grouping is question-disjoint and gold-blind."""
import json
import pytest

from scripts.train_action_sft import split_training_rows


def fake_rows(n=30):
    rows=[]
    for i in range(n):
        for step in range(3):
            rows.append({
                "prompt":[{"role":"system","content":"Search with tools"},
                          {"role":"user","content":json.dumps({
                              "question":f"question {i}","tool_history":[],
                              "remaining_steps":5})}],
                "completion":[{"role":"assistant","content":'{"kind":"search","query":"A"}'}],
                "question_id":f"train-{i}",
                "step":step,
            })
    return rows


def test_split_training_action_rows_by_question_not_by_step():
    rows=fake_rows()
    train,validation=split_training_rows(rows,seed=43,max_examples=90,
                                         validation_percentage=25)
    t={r["question_id"] for r in train}
    v={r["question_id"] for r in validation}
    assert t and v and not (t&v)
    t2,v2=split_training_rows(list(reversed(rows)),seed=43,max_examples=90,
                              validation_percentage=25)
    assert {r["question_id"] for r in t2}==t
    assert {r["question_id"] for r in v2}==v


def test_training_prompt_with_gold_key_fails_closed():
    rows=fake_rows()
    rows[0]["prompt"][1]["content"]='{"question":"q","gold_answer":"secret"}'
    with pytest.raises(ValueError,match="ground truth"):
        split_training_rows(rows,seed=43,max_examples=90,
                            validation_percentage=25)


def test_invalid_train_sample_shape_rejected():
    rows=fake_rows()
    rows[0]["evaluation_gold_answer"]="secret"
    with pytest.raises(ValueError,match="unexpected training row fields"):
        split_training_rows(rows,seed=43,max_examples=90,
                            validation_percentage=25)
