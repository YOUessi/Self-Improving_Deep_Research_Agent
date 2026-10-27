"""TRAIN answer-ceiling diagnostic cannot inspect frozen DEV and labels stay offline."""
import json

import pytest

from scripts.audit_train_finite_answer_ceiling import (
    audit,evaluate_source_exposure
)
from sidra.io import save_jsonl
from sidra.schema import Document,Question


def test_boolean_answer_is_visible_but_not_in_finite_action_catalog():
    docs={
        "a":Document("a","Alpha","The statement is yes, according to Alpha."),
        "b":Document("b","Beta","A second supporting fact says yes again."),
    }
    q=Question("train-q","Is it true?","yes",
               supporting_doc_ids=("a","b"),split="train")
    result=evaluate_source_exposure(q,docs,proposer="v4")
    assert result["answer_visible"]
    assert result["gold_is_boolean"]
    assert not result["offered_in_final_actions"]
    assert result["number_of_final_candidates"]<=4
    with pytest.raises(ValueError,match="TRAIN"):
        evaluate_source_exposure(Question("dev","Is it true?","yes",
            supporting_doc_ids=("a","b"),split="dev"),docs)


def test_train_only_dataset_audit_and_immutability(tmp_path):
    data=tmp_path/"train"/"converted"
    data.mkdir(parents=True)
    docs=[Document("a","Alpha","The best answer is 1957."),
          Document("b","Beta","Alpha is discussed in 1957.")]
    q=Question("train-q","In which year?","1957",
               supporting_doc_ids=("a","b"),split="train")
    save_jsonl(data/"documents.jsonl",[d.__dict__ for d in docs])
    save_jsonl(data/"questions.jsonl",[q.__dict__])
    out=tmp_path/"ceiling.json"
    stats=audit(data,out)
    assert stats["n"]==1
    assert stats["uses_train_gold_for_OFFLINE_exposure_and_scoring"]
    assert not stats["gold_exposed_to_model_or_online_policy"]
    assert "gold_answer" not in out.read_text()
    with pytest.raises(FileExistsError):audit(data,out)
    frozen=tmp_path/"final_b160"/"converted"
    frozen.mkdir(parents=True)
    with pytest.raises(ValueError,match="TRAIN"):
        audit(frozen,tmp_path/"should_not_exist")


def test_train_only_audit_rejects_bad_split_even_when_path_says_train(tmp_path):
    data=tmp_path/"train"/"converted"
    data.mkdir(parents=True)
    save_jsonl(data/"documents.jsonl",[Document("a","A","yes").__dict__])
    save_jsonl(data/"questions.jsonl",[
        Question("dev","Q","yes",supporting_doc_ids=("a",),split="dev").__dict__])
    with pytest.raises(ValueError,match="DEV/TEST"):
        audit(data,tmp_path/"blocked.json")
