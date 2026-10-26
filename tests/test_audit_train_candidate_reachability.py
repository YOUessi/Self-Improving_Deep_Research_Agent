"""Offline answer-proposal upper-bound audit is TRAIN-only and does not train."""
from __future__ import annotations

import json

import pytest
from sidra.io import save_jsonl
from scripts.audit_train_candidate_reachability import audit_train_only


def _train_fixture(parent,*,split="train"):
    source=parent/"dataset"
    source.mkdir()
    save_jsonl(source/"documents.jsonl",[
        {"doc_id":"q1::0","scope":"q1","title":"Congress of Vienna",
         "text":"The Congress was chaired by Austrian statesman Klemens von Metternich."},
        {"doc_id":"q1::1","scope":"q1","title":"Duchy of Genoa",
         "text":"The Duchy of Genoa was assigned in Vienna."},
    ])
    save_jsonl(source/"questions.jsonl",[
        {"question_id":"q1","question":"Which Austrian statesman chaired the meeting?",
         "gold_answer":"Klemens von Metternich",
         "supporting_doc_ids":["q1::0","q1::1"],"scope":"q1",
         "split":split}
    ])
    return source


def test_frozen_train_aggregate_no_labels_in_output(tmp_path):
    source=_train_fixture(tmp_path)
    out=tmp_path/"output.json"
    stats=audit_train_only(source,out,seed=8)
    assert stats["train_questions"]==1
    assert stats["gold_used_in_offline_diagnostic"] is True
    assert stats["gold_used_by_actual_agent"] is False
    assert stats["model_learning_demonstrated"] is False
    assert stats["metrics"]["v2_candidate_answer_available"]["count"]==1
    text=out.read_text()
    assert "Klemens" not in text
    assert "Which Austrian" not in text
    assert "q1::0" not in text
    assert json.loads(text)["stage"]==stats["stage"]
    with pytest.raises(FileExistsError):
        audit_train_only(source,out,seed=8)


def test_offline_oracle_audit_cannot_use_dev_test(tmp_path):
    for split in ("dev","test"):
        root=tmp_path/split
        root.mkdir()
        data=_train_fixture(root,split=split)
        with pytest.raises(ValueError,match="TRAIN-only"):
            audit_train_only(data,root/"rejected.json")
        assert not (root/"rejected.json").exists()
